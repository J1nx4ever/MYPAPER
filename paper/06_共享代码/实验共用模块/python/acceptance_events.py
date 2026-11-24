"""Independent replay of opt-in AnyLogic boundary-event sidecars.

This module never starts a simulator, server or training job. Event time is the
observed callback time; FIFO processing has its own physical interval. Complete
trace integrity is necessary evidence, not proof that all GUI hooks were added.
"""
from collections import deque
import json
import math
from pathlib import Path

from .hdf5_events import EventStorageError, iter_event_bundle


class EventEvidenceError(ValueError):
    """Missing, corrupt, incomplete or internally inconsistent evidence."""


EVENT_TYPES = frozenset({
    'EPISODE_START', 'WAIT_ENTER', 'WAIT_EXIT', 'FIFO_ENQUEUE', 'FIFO_PROCESS',
    'FIFO_DROP', 'WINDOW_CLOSE', 'LOAD_COMPLETE', 'UNLOAD_COMPLETE',
    'CUTOFF_EXCLUDED', 'DECISION', 'ACTION_APPLIED', 'TERMINAL_REQUEST', 'EPISODE_END',
})
REQUIRED_CONFIG = ('shift_seconds', 'window_seconds', 'grade_lower', 'grade_upper',
                   'grade_epsilon', 'tonnage_epsilon', 'mass_balance_epsilon',
                   'crusher_rate_tph')


def _require(condition, message):
    if not condition:
        raise EventEvidenceError(message)


def _number(value, name, *, minimum=None):
    _require(type(value) in (float, int) and math.isfinite(value), f'invalid {name}')
    if minimum is not None:
        _require(value >= minimum, f'negative/out-of-range {name}')
    return float(value)


def _integer(value, name, *, minimum=0):
    _require(type(value) is int and value >= minimum, f'invalid {name}')
    return value


def _hash(value, name):
    _require(isinstance(value, str) and len(value) == 64 and
             all(c in '0123456789abcdef' for c in value), f'invalid {name}')
    return value


def _close(actual, expected, *, eps, name):
    _require(abs(actual - expected) <= eps, f'{name}: actual={actual} expected={expected}')


def _wait_close(actual, expected, name):
    # Same numerical comparison as the existing transition adapter. This is an
    # audit comparison tolerance, not a change to any simulation epsilon.
    _require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7),
             f'{name}: actual={actual} expected={expected}')


def _wait_key(data):
    _require(data.get('resource') in ('load', 'dump'), 'invalid wait resource')
    return (_integer(data.get('truck_id'), 'truck_id'), data['resource'],
            _integer(data.get('site_id'), 'site_id'))


def _material(data):
    tons = _number(data.get('tons'), 'tons', minimum=0)
    metal = _number(data.get('metal_tons'), 'metal_tons', minimum=0)
    _require(metal <= tons + 1e-12, 'metal exceeds gross tons')
    return tons, metal


def _json_object(path):
    try:
        data = _strict_json(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise EventEvidenceError(f'cannot read evidence {path.name}: {exc}') from exc
    _require(type(data) is dict, f'{path.name} must contain a JSON object')
    return data


def _strict_json(value):
    def pairs(items):
        result = {}
        for key, item in items:
            _require(key not in result, f'duplicate JSON field: {key}')
            result[key] = item
        return result
    def constant(item):
        raise EventEvidenceError(f'nonfinite JSON constant: {item}')
    return json.loads(value, object_pairs_hook=pairs, parse_constant=constant)


def replay_event_bundle(bundle_dir, *, expected_episode_id=None,
                        expected_config_sha256=None, expected_source_kind=None,
                        expected_recording_id=None, transition_checks=None):
    """Stream a JSONL, gzip JSONL or typed HDF5 bundle; reject incomplete traces.

    ``config_sha256`` identifies the exact acceptance_run_config.json bytes, not
    RewardConfiguration.fingerprint. The caller must supply the expected hash to
    establish a trusted join with a separate transition recording. Returned
    ``requests`` retain identities for that join. Synthetic input stays labelled
    synthetic; passing this verifier never grants collection/training approval.
    """
    directory = Path(bundle_dir)
    manifest = _json_object(directory / 'events_manifest.json')
    _require(manifest.get('schema_version') == 1 and
             manifest.get('record_kind') == 'anylogic_boundary_events', 'unsupported event manifest')
    _require(manifest.get('complete') is True and manifest.get('terminated') is True
             and manifest.get('truncated') is False, 'event recording is incomplete or abnormal')
    source = manifest.get('source_kind')
    _require(source in ('actual_anylogic', 'synthetic_unit_test'), 'unknown event source')
    if expected_source_kind is not None:
        _require(source == expected_source_kind, 'event source does not match expected source')
    episode = manifest.get('episode_id')
    _require(isinstance(episode, str) and episode, 'missing episode identity')
    config_hash = _hash(manifest.get('config_sha256'), 'config_sha256')
    if expected_episode_id is not None:
        _require(episode == expected_episode_id, 'event/transition episode identity mismatch')
    if expected_config_sha256 is not None:
        _require(config_hash == expected_config_sha256, 'event/transition config hash mismatch')
    _integer(manifest.get('seed'), 'seed')
    _integer(manifest.get('run_index'), 'run_index', minimum=1)
    for key in ('recording_id', 'instrumentation_revision'):
        _require(isinstance(manifest.get(key), str) and manifest[key], f'missing {key}')
    if expected_recording_id is not None:
        _require(manifest['recording_id'] == expected_recording_id, 'recording identity mismatch')
    config = manifest.get('config')
    _require(type(config) is dict, 'missing event configuration')
    for key in REQUIRED_CONFIG:
        _number(config.get(key), key, minimum=0)
    _require(config['shift_seconds'] == 28800 and config['window_seconds'] == 7200,
             'unsupported shift/window duration')
    _require(0 <= config['grade_lower'] <= config['grade_upper'] <= 1
             and config['crusher_rate_tph'] > 0, 'invalid grade bounds or crusher rate')
    cutoff, width = config['shift_seconds'], config['window_seconds']
    _close(_number(manifest.get('cutoff_seconds'), 'cutoff_seconds'), cutoff, eps=0,
           name='manifest cutoff')
    event_count = _integer(manifest.get('event_count'), 'event_count', minimum=2)

    def events():
        # The iterator verifies its final hash/count when exhausted.  No replay
        # result can be returned until this loop finishes successfully.
        try:
            yield from iter_event_bundle(directory, manifest)
        except EventStorageError as exc:
            raise EventEvidenceError(str(exc)) from exc

    waits, wait_intervals = {}, []
    cumulative_wait = 0.
    fifo, batches = deque(), {}
    payloads, payload_ids = {}, set()
    windows, requests, critical_same_time = [], [], []
    output_tons, output_metal, waste_tons = 0., 0., 0.
    enqueue_tons, enqueue_metal, dropped_tons, dropped_metal = 0., 0., 0., 0.
    last_process_end, last_time, terminal_seen = 0., -1., False
    last_request_wait, last_request_sequence = 0., -1
    fallback_entries, excluded, last_critical = 0, [], None
    mass_eps = config['mass_balance_epsilon']
    critical = {'FIFO_ENQUEUE', 'FIFO_PROCESS', 'WINDOW_CLOSE', 'DECISION',
                'TERMINAL_REQUEST', 'LOAD_COMPLETE', 'UNLOAD_COMPLETE', 'CUTOFF_EXCLUDED'}
    totals = [[0., 0.] for _ in range(4)]

    for seq, event in enumerate(events()):
        _require(type(event) is dict and event.get('schema_version') == 1,
                 f'invalid event envelope at {seq}')
        _require(event.get('episode_id') == episode and event.get('config_sha256') == config_hash,
                 f'event identity changed at {seq}')
        _require(type(event.get('event_seq')) is int and event['event_seq'] == seq,
                 f'event sequence gap/duplicate at {seq}')
        now = _number(event.get('sim_time_seconds'), 'sim_time_seconds', minimum=0)
        _require(last_time <= now <= cutoff, 'observed event time regressed or exceeds cutoff')
        last_time = now
        kind, data = event.get('event_type'), event.get('data')
        _require(kind in EVENT_TYPES and type(data) is dict, f'unknown event type/data at {seq}')
        _require((kind == 'EPISODE_START') == (seq == 0), 'start must be first and unique')
        _require((kind == 'EPISODE_END') == (seq == event_count-1),
                 'event count/end mismatch: end must be last and unique')
        if terminal_seen:
            _require(kind == 'EPISODE_END', 'event after terminal request')
        if kind in critical:
            if last_critical is not None and last_critical['sim_time_seconds'] == now:
                critical_same_time.append(dict(before=last_critical['event_type'], after=kind,
                    before_seq=last_critical['event_seq'], after_seq=seq, sim_time_seconds=now))
            last_critical = event

        if kind == 'EPISODE_START':
            _require(now == 0. and data.get('initial_fifo') == []
                     and data.get('initial_waits') == [] and data.get('initial_payloads') == [],
                     'recording must begin at empty/idle shift initialization')
            _require(data.get('config') == config and data.get('seed') == manifest['seed']
                     and data.get('run_index') == manifest['run_index'], 'start/manifest config mismatch')
        elif kind == 'WAIT_ENTER':
            key = _wait_key(data)
            _require(not any(k[0] == key[0] for k in waits), 'overlapping wait for one truck')
            _require(type(data.get('fallback')) is bool, 'missing fallback marker')
            engine_start = _number(data.get('engine_queue_start_seconds'), 'engine_queue_start_seconds', minimum=0)
            _wait_close(engine_start, now, 'wait entry timestamp/unit mismatch')
            waits[key] = (now, seq, data['fallback'])
            fallback_entries += int(data['fallback'])
        elif kind == 'WAIT_EXIT':
            key = _wait_key(data)
            _require(key in waits, 'wait exit without entry')
            start, enter_seq, fallback = waits.pop(key)
            _require(data.get('fallback') is fallback, 'fallback marker changed')
            duration = now-start
            _wait_close(_number(data.get('engine_wait_seconds'), 'engine_wait_seconds', minimum=0),
                        duration, 'engine/independent wait duration mismatch')
            cumulative_wait += duration
            wait_intervals.append(dict(truck_id=key[0], resource=key[1], site_id=key[2],
                start_seconds=start, end_seconds=now, seconds=duration,
                enter_seq=enter_seq, exit_seq=seq, fallback=fallback, active_at_cutoff=False))
        elif kind == 'LOAD_COMPLETE':
            truck = _integer(data.get('truck_id'), 'truck_id')
            payload = data.get('payload_id')
            _require(isinstance(payload, str) and payload and payload not in payload_ids,
                     'missing/duplicate payload identity')
            _require(truck not in payloads and now < cutoff, 'duplicate or cutoff load completion')
            tons, metal = _material(data)
            _require(tons > 0 and data.get('material') in ('ore', 'waste'), 'invalid loaded payload')
            if data['material'] == 'waste':
                _close(metal, 0., eps=mass_eps, name='waste metal')
            payload_ids.add(payload)
            payloads[truck] = dict(payload_id=payload, material=data['material'], tons=tons, metal_tons=metal)
        elif kind == 'UNLOAD_COMPLETE':
            truck = _integer(data.get('truck_id'), 'truck_id')
            _require(truck in payloads and now < cutoff, 'unload without load or at cutoff')
            payload = payloads.pop(truck)
            _require(data.get('payload_id') == payload['payload_id'] and
                     data.get('material') == payload['material'], 'unload payload identity mismatch')
            tons, metal = _material(data)
            _close(tons, payload['tons'], eps=mass_eps, name='loaded/unloaded tons')
            _close(metal, payload['metal_tons'], eps=mass_eps, name='loaded/unloaded metal')
            if data['material'] == 'waste':
                waste_tons += tons
            else:
                _require(any(b['payload_id'] == data['payload_id'] for b in batches.values()),
                         'ore unload did not enqueue its FIFO batch')
        elif kind == 'FIFO_ENQUEUE':
            batch_id = data.get('batch_id')
            _require(isinstance(batch_id, str) and batch_id and batch_id not in batches,
                     'missing/duplicate FIFO identity')
            tons, metal = _material(data)
            _require(tons > 0 and now < cutoff, 'invalid FIFO enqueue or enqueue at cutoff')
            batch = dict(batch_id=batch_id, tons=tons, metal_tons=metal, initial_tons=tons,
                         initial_metal=metal, enqueued_at=now, payload_id=data.get('payload_id'),
                         processed_windows=[])
            _require(isinstance(batch['payload_id'], str) and batch['payload_id'] in payload_ids,
                     'FIFO batch has no recorded loaded payload')
            _require(not any(b['payload_id'] == batch['payload_id'] for b in batches.values()),
                     'payload entered FIFO twice')
            payload = payloads.get(_integer(data.get('truck_id'), 'truck_id'))
            _require(payload is not None and payload['material'] == 'ore'
                     and payload['payload_id'] == batch['payload_id'], 'FIFO payload is not in-transit ore')
            _close(tons, payload['tons'], eps=mass_eps, name='unloading/FIFO tons')
            _close(metal, payload['metal_tons'], eps=mass_eps, name='unloading/FIFO metal')
            fifo.append(batch)
            batches[batch_id] = batch
            enqueue_tons += tons
            enqueue_metal += metal
        elif kind == 'FIFO_PROCESS':
            _require(fifo and data.get('batch_id') == fifo[0]['batch_id'], 'FIFO oldest batch was bypassed')
            batch = fifo[0]
            j = _integer(data.get('window_index'), 'window_index')
            _require(j == len(windows) and j < 4, 'processing belongs to closed/wrong window')
            start = _number(data.get('interval_start_seconds'), 'interval_start_seconds', minimum=0)
            end = _number(data.get('interval_end_seconds'), 'interval_end_seconds', minimum=0)
            _require(start >= max(last_process_end, batch['enqueued_at'])-1e-8 and
                     j*width-1e-8 <= start < end <= (j+1)*width+1e-8 and end <= now+1e-8,
                     'invalid/overlapping/cross-window FIFO interval')
            _wait_close(start, max(last_process_end, batch['enqueued_at']),
                        'unexplained idle gap while FIFO has ore')
            tons, metal = _material(data)
            _require(tons > 0 and tons <= batch['tons']+mass_eps, 'invalid FIFO consumption')
            _close(tons, (end-start)*config['crusher_rate_tph']/3600., eps=mass_eps,
                   name='physical crusher time/rate/tons')
            _close(metal, tons*batch['initial_metal']/batch['initial_tons'], eps=mass_eps,
                   name='FIFO processed metal')
            batch['tons'] -= tons
            batch['metal_tons'] -= metal
            if j not in batch['processed_windows']:
                batch['processed_windows'].append(j)
            totals[j][0] += tons
            totals[j][1] += metal
            output_tons += tons
            output_metal += metal
            last_process_end = end
        elif kind == 'FIFO_DROP':
            _require(fifo and data.get('batch_id') == fifo[0]['batch_id'], 'FIFO drop must remove head')
            batch = fifo.popleft()
            tons, metal = _material(data)
            _require(data.get('reason') == 'numerical_tail_le_1e-9' and tons <= 1e-9
                     and batch['tons'] >= -1e-12 and batch['tons'] <= 1e-9,
                     'FIFO removal exceeded current numerical-tail rule')
            _close(tons, max(0., batch['tons']), eps=1e-12, name='FIFO dropped tons')
            # Reconstruct the tiny discarded tail using its original grade;
            # accumulated subtraction error in an independently tracked metal
            # sum must not be mistaken for a physical extra tail.
            _close(metal, tons*batch['initial_metal']/batch['initial_tons'],
                   eps=1e-12, name='FIFO dropped metal')
            dropped_tons += tons
            dropped_metal += metal
        elif kind == 'WINDOW_CLOSE':
            j = _integer(data.get('window_index'), 'window_index')
            _require(j == len(windows) and j < 4 and now >= (j+1)*width,
                     'duplicate/out-of-order/early window close')
            _close(_number(data.get('window_end_seconds'), 'window_end_seconds'), (j+1)*width,
                   eps=0, name='window boundary')
            tons, metal = _material(data)
            _close(tons, totals[j][0], eps=mass_eps, name='window FIFO tons')
            _close(metal, totals[j][1], eps=mass_eps, name='window FIFO metal')
            grade = metal/tons if tons > 0 else 0.
            _close(_number(data.get('grade'), 'grade'), grade, eps=max(config['grade_epsilon'], 1e-15),
                   name='window grade')
            lower = tons <= 0 or grade < config['grade_lower']-config['grade_epsilon']
            upper = tons > 0 and grade > config['grade_upper']+config['grade_epsilon']
            _require(type(data.get('violation_lower')) is bool and data['violation_lower'] == lower
                     and type(data.get('violation_upper')) is bool and data['violation_upper'] == upper
                     and type(data.get('cost')) is int and data['cost'] == int(lower or upper),
                     'window violation label mismatch')
            windows.append(dict(event_seq=seq, observed_time_seconds=now, window_index=j,
                ore_tons=tons, metal_tons=metal, grade=grade, cost=int(lower or upper)))
        elif kind == 'CUTOFF_EXCLUDED':
            _require(now == cutoff and data.get('operation') in ('loading', 'unloading', 'crusher_receive'),
                     'invalid cutoff exclusion')
            _integer(data.get('truck_id'), 'truck_id')
            excluded.append(dict(event_seq=seq, **data))
        elif kind in ('DECISION', 'TERMINAL_REQUEST'):
            _require(not requests or requests[-1]['terminal'] or requests[-1]['action'] is not None,
                     'new request before action application was recorded')
            sequence = _integer(data.get('decision_sequence'), 'decision_sequence')
            _require(sequence == last_request_sequence+1 and
                     data.get('request_id') == f'{episode}-{sequence}', 'request identity/sequence gap')
            current = cumulative_wait + sum(now-start for start, _, _ in waits.values())
            _wait_close(_number(data.get('cumulative_wait_seconds'), 'cumulative_wait_seconds', minimum=0),
                        current, 'independent cumulative waiting mismatch')
            delta = current-last_request_wait
            _wait_close(_number(data.get('delta_wait_seconds'), 'delta_wait_seconds', minimum=0),
                        delta, 'independent interval waiting mismatch')
            if kind == 'DECISION':
                _require(now < cutoff, 'invalid decision time')
            else:
                _require(now == cutoff and len(windows) == 4, 'terminal lacks cutoff/four windows')
                terminal_seen = True
            requests.append(dict(event_seq=seq, request_id=data['request_id'], decision_sequence=sequence,
                sim_time_seconds=now, cumulative_wait_seconds=current, delta_wait_seconds=delta,
                action=None, terminal=kind == 'TERMINAL_REQUEST',
                window_indices=[w['window_index'] for w in windows]))
            last_request_wait, last_request_sequence = current, sequence
        elif kind == 'ACTION_APPLIED':
            _require(requests and not requests[-1]['terminal'] and requests[-1]['action'] is None,
                     'duplicate action application or no pending request')
            pending = requests[-1]
            _require(data.get('request_id') == pending['request_id'] and
                     data.get('decision_sequence') == pending['decision_sequence'] and
                     data.get('action') in (0, 1, 2, 3) and type(data['action']) is int and
                     now == pending['sim_time_seconds'], 'action application identity/time mismatch')
            pending['action'] = data['action']
            pending['action_event_seq'] = seq
        elif kind == 'EPISODE_END':
            _require(terminal_seen and now == cutoff and len(windows) == 4,
                     'end missing acknowledged terminal/four windows')
            for (truck, resource, site), (start, enter_seq, fallback) in waits.items():
                duration = cutoff-start
                cumulative_wait += duration
                wait_intervals.append(dict(truck_id=truck, resource=resource, site_id=site,
                    start_seconds=start, end_seconds=cutoff, seconds=duration,
                    enter_seq=enter_seq, exit_seq=None, fallback=fallback, active_at_cutoff=True))
            checks = {'cumulative_wait_seconds': cumulative_wait, 'ore_tons': output_tons,
                      'metal_tons': output_metal, 'waste_tons': waste_tons,
                      'fifo_tons': sum(b['tons'] for b in fifo),
                      'fifo_metal_tons': sum(b['metal_tons'] for b in fifo)}
            for key, expected in checks.items():
                actual = _number(data.get(key), key, minimum=0)
                if key == 'cumulative_wait_seconds':
                    _wait_close(actual, expected, 'terminal independent waiting')
                else:
                    _close(actual, expected, eps=mass_eps, name='terminal '+key)
            _require(data.get('terminated') is True and data.get('truncated') is False,
                     'terminal flags are not normal shift completion')

    _require(requests and requests[0]['decision_sequence'] == 0 and
             requests[0]['cumulative_wait_seconds'] == 0, 'missing/unaccounted first decision')
    if fifo:
        _wait_close(last_process_end, cutoff, 'nonempty FIFO was not integrated to cutoff')
    _close(enqueue_tons, output_tons+sum(b['tons'] for b in fifo)+dropped_tons,
           eps=mass_eps, name='FIFO shift tons conservation')
    _close(enqueue_metal, output_metal+sum(b['metal_tons'] for b in fifo)+dropped_metal,
           eps=mass_eps, name='FIFO shift metal conservation')
    _require(all(b['payload_id'] not in {p['payload_id'] for p in payloads.values()}
                 for b in batches.values()), 'FIFO receipt lacks matching unload completion')
    if transition_checks is not None:
        _require(len(transition_checks) == len(requests), 'transition/event request count mismatch')
        for expected, observed in zip(transition_checks, requests):
            for key in ('request_id', 'decision_sequence', 'sim_time_seconds', 'action'):
                _require(expected.get(key) == observed[key], f'transition/event {key} mismatch')
            for key in ('cumulative_wait_seconds', 'delta_wait_seconds'):
                _wait_close(_number(expected.get(key), key), observed[key], 'transition/event '+key)
    closed_counts = [len(r['window_indices']) for r in requests]
    return dict(schema_version=1, source_kind=source, episode_id=episode,
        recording_id=manifest['recording_id'],instrumentation_revision=manifest['instrumentation_revision'],
        seed=manifest['seed'],run_index=manifest['run_index'],config=config,
        config_sha256=config_hash, events_sha256=manifest['events_sha256'],
        status='complete_event_replay_requires_review', event_count=event_count,
        independent_wait_seconds=cumulative_wait, wait_intervals=wait_intervals,
        requests=requests, decisions=requests, windows=windows, ore_tons=output_tons, metal_tons=output_metal,
        waste_tons=waste_tons, fifo_remaining=list(fifo), fifo_dropped_tons=dropped_tons,
        fifo_dropped_metal_tons=dropped_metal, in_transit_payloads=payloads,
        material_totals=dict(ore_tons=output_tons,metal_tons=output_metal,waste_tons=waste_tons,
            fifo_tons=sum(b['tons'] for b in fifo),fifo_metal_tons=sum(b['metal_tons'] for b in fifo),
            enqueued_tons=enqueue_tons,enqueued_metal_tons=enqueue_metal,
            dropped_tons=dropped_tons,dropped_metal_tons=dropped_metal),
        observed_same_timestamp_order=critical_same_time, cutoff_exclusions=excluded,
        coverage=dict(empty_window=any(w['ore_tons'] == 0 for w in windows),
            multi_window_between_requests=any(b-a > 1 for a, b in zip(closed_counts, closed_counts[1:])),
            same_timestamp_critical_events=bool(critical_same_time),
            cutoff_exclusion=bool(excluded), wait_remaining_at_cutoff=bool(waits),
            fallback_wait_path=fallback_entries > 0,
            fifo_spans_windows=any(len(b['processed_windows']) > 1 for b in batches.values())),
        limitations=['Hash/sequence checks cannot establish that omitted GUI hooks never occurred.',
            'Observed same-timestamp order is this trace order, not a universal order implied by LIFO.',
            'Boundary coverage needs reviewed scenario inputs; one normal run is not full acceptance.'])
