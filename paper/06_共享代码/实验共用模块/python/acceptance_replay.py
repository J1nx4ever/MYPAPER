"""Read-only independent replay of opt-in boundary acceptance recordings.

No server, simulator, dataset collector or learner is constructed here. Derived
MC targets are audit results, never a training dataset or acceptance certificate.
Deterministically reserialized Protobuf messages, actual float32 inputs, actions
and settlement masks are cross-checked before any result is returned. Stored
payloads preserve decoded message contents; they are not captured network frames.
Event sidecars supply the separate physical evidence needed to reconstruct
waiting and FIFO flow independently.
"""
import argparse
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from google.protobuf.message import DecodeError

from .generated import dispatch_pb2 as pb
from .observation_contract import FIELD_NAMES, OBSERVATION_SCHEMA_ID, validate_observation
from .reward_config import configuration_from_components
from .acceptance_scenario import validate_scenario_jobs, validate_scenario_rpc
from .reward_contract import (REWARD_CONTRACT_ID, validate_reward_accounting,
                              validate_window_summary, window_summary_to_dict)


class RecordingReplayError(ValueError):
    """An incomplete, corrupt or contradictory recording is not replayable."""


def _require(condition, message):
    if not condition:
        raise RecordingReplayError(message)


def _close(actual, expected, name, *, absolute=1e-9):
    _require(type(actual) in (int, float) and math.isfinite(actual) and
             math.isclose(actual, expected, rel_tol=1e-10, abs_tol=absolute),
             f'{name}: actual={actual} expected={expected}')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _object(path):
    try:
        result = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise RecordingReplayError(f'cannot read {path.name}: {exc}') from exc
    _require(type(result) is dict, f'{path.name} must be an object')
    return result


def _lines(directory, manifest, name):
    metadata = manifest.get('files', {}).get(name)
    _require(type(metadata) is dict, f'missing file metadata: {name}')
    try:
        raw = (directory / name).read_bytes()
    except OSError as exc:
        raise RecordingReplayError(f'missing stream {name}: {exc}') from exc
    _require(raw.endswith(b'\n'), f'incomplete final line: {name}')
    _require(_sha(raw) == metadata.get('sha256'), f'file hash mismatch: {name}')
    lines = raw.splitlines()
    _require(type(metadata.get('records')) is int and len(lines) == metadata['records'],
             f'file record count mismatch: {name}')
    rows = []
    for sequence, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            raise RecordingReplayError(f'invalid JSON: {name} row {sequence}') from exc
        _require(type(row) is dict and type(row.get('record_seq')) is int and
                 row['record_seq'] == sequence, f'noncontiguous record_seq: {name}')
        rows.append(row)
    return rows


def _payload(row, message_type):
    try:
        raw = base64.b64decode(row.get('payload_base64', ''), validate=True)
        _require(raw and _sha(raw) == row.get('payload_sha256'), 'wire payload hash mismatch')
        message = message_type.FromString(raw)
    except (ValueError, TypeError, DecodeError) as exc:
        raise RecordingReplayError(f'invalid wire payload: {exc}') from exc
    _require(message.episode_id == row.get('episode_id'), 'wire/record episode identity mismatch')
    return raw, message


def _wire_info(request):
    """Explicit wire-to-info comparison, independent of the environment adapter."""
    return dict(action_mask=[bool(v) for v in request.obs.values[19:23]],
        episode_id=request.episode_id, truck_id=int(request.truck_id),
        decision_sequence=int(request.decision_sequence), request_id=request.request_id,
        sim_time_seconds=request.sim_time_seconds, sim_time_delta_seconds=request.sim_time_delta_seconds,
        previous_action=int(request.previous_action), transition_reward_raw=request.transition_reward,
        reward_components=dict(request.reward_components),
        closed_windows=[window_summary_to_dict(w) for w in request.closed_windows],
        experiment_run_index=int(request.experiment_run_index), experiment_seed=int(request.experiment_seed),
        reward_contract_id=request.reward_contract_id, observation_schema_id=request.obs.schema_id)


def _observation(values, request, name):
    try:
        arr = validate_observation(request.obs.schema_id, values,
            terminal=request.kind == pb.EPISODE_END, truck_id=request.truck_id)
    except (ValueError, TypeError) as exc:
        raise RecordingReplayError(f'{name}: {exc}') from exc
    # Records are the tensors actually seen by the rule/policy, not a rounded
    # reconstruction or an unapproved projection appended to the wire state.
    _require(np.array_equal(np.asarray(values, dtype=np.float64), arr.astype(np.float64)),
             f'{name} is not exact float32 input')
    _require(np.array_equal(arr, np.asarray(request.obs.values, dtype=np.float32)),
             f'{name} differs from raw wire input')
    return arr


def _wire_stream(rows):
    requests, responses, unique = {}, {}, []
    request_counts, response_counts = Counter(), Counter()
    for row in rows:
        kind = row.get('event')
        _require(kind in ('request', 'response', 'rejected'), 'unknown RPC recording event')
        _require(kind != 'rejected', 'recording contains rejected RPC; not a complete normal shift')
        raw, message = _payload(row, pb.DispatchRequest if kind == 'request' else pb.DispatchResponse)
        identity = row.get('request_id')
        _require(isinstance(identity, str) and identity, 'missing RPC request identity')
        if kind == 'request':
            request_counts[identity] += 1
            _require(message.request_id == identity, 'wire/record request identity mismatch')
            if identity in requests:
                _require(requests[identity][0] == raw, 'request_id reused with different wire content')
            else:
                requests[identity] = (raw, message)
                unique.append(identity)
        else:
            response_counts[identity] += 1
            _require(identity in requests, 'response appears before its request')
            request = requests[identity][1]
            _require(message.episode_id == request.episode_id and
                     message.decision_sequence == request.decision_sequence,
                     'response lifecycle correlation mismatch')
            expected = (bool(message.has_action), int(message.action))
            if identity in responses:
                _require(responses[identity] == expected, 'duplicate response action changed')
            responses[identity] = expected
            if request.action_required:
                _require(message.has_action and 0 <= message.action < 4 and
                         request.obs.values[19+message.action] == 1, 'illegal wire action')
            else:
                _require(not message.has_action, 'terminal response must be an ACK')
    _require(requests and set(requests) == set(responses), 'missing wire request/response pairs')
    _require(request_counts == response_counts, 'missing/extra RPC retry response counts')
    return {k: v[1] for k, v in requests.items()}, responses, unique


def _info(row_info, request, name):
    _require(row_info == _wire_info(request), f'{name} differs from original wire fields')


def _runtime(request, cfg, baseline):
    terminal = request.kind == pb.EPISODE_END
    try:
        validate_scenario_rpc(cfg,dict(request.reward_components))
        runtime = configuration_from_components(dict(request.reward_components), terminal=terminal)
    except (ValueError, KeyError, TypeError) as exc:
        raise RecordingReplayError(f'invalid runtime configuration: {exc}') from exc
    _require(baseline is None or baseline == runtime, 'immutable runtime configuration changed')
    for remote, local in (('wait_kappa', 'kappa'), ('w_production', 'ore_weight'),
                         ('waste_shortfall_weight', 'waste_weight'), ('grade_lower', 'grade_lower'),
                         ('grade_upper', 'grade_upper'), ('grade_epsilon', 'grade_epsilon'),
                         ('tonnage_epsilon', 'tonnage_epsilon'), ('mass_balance_epsilon', 'mass_balance_epsilon')):
        _require(runtime.get(remote) == cfg.get('reward', {}).get(local),
                 f'RPC/config reward setting mismatch: {local}')
    schema_hash = _sha(('\n'.join(FIELD_NAMES)+'\n').encode())
    for j in range(8):
        _require(request.reward_components.get(f'config.obs.schema_word_{j}') ==
                 int(schema_hash[j*8:j*8+8], 16), 'wire observation field-order hash mismatch')
    _require(request.reward_contract_id == REWARD_CONTRACT_ID and
             request.obs.schema_id == OBSERVATION_SCHEMA_ID, 'wire contract mismatch')
    return runtime


def _material(components, windows, cfg):
    """Independent shift totals check; event sidecars reconstruct the raw flow."""
    keys = ('ore_tons_shift', 'waste_tons_shift', 'crusher_received_tons', 'crusher_buffer_tons',
            'mined_ore_tons', 'mined_waste_tons', 'carried_ore_tons', 'carried_waste_tons',
            'load_sites_reserved_tons', 'assigned_not_loaded_tons', 'shift_grade')
    values = {}
    for key in keys:
        value = components.get(key)
        _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                 f'missing/invalid terminal material: {key}')
        values[key] = value
    v = values
    residuals = dict(
        window_ore=sum(w['ore_tons'] for w in windows)-v['ore_tons_shift'],
        crusher_ore=v['crusher_received_tons']-v['ore_tons_shift']-v['crusher_buffer_tons'],
        mined_ore=v['mined_ore_tons']-v['crusher_received_tons']-v['carried_ore_tons'],
        mined_waste=v['mined_waste_tons']-v['waste_tons_shift']-v['carried_waste_tons'],
        reserved=v['load_sites_reserved_tons']-v['assigned_not_loaded_tons'],
        window_metal=sum(w['metal_tons'] for w in windows)-v['ore_tons_shift']*v['shift_grade'])
    _require(v['shift_grade'] <= 1 and all(abs(x) <= cfg.get('mass_balance_epsilon')
             for x in residuals.values()), 'terminal material reconciliation failed')
    return residuals


def _request_lifecycle(request, *, sequence, now, elapsed, terminal):
    _require(request.decision_sequence == sequence and request.kind ==
             (pb.EPISODE_END if terminal else (pb.EPISODE_START if sequence == 0 else pb.DECISION)),
             'wire decision lifecycle/sequence mismatch')
    _require(request.terminated is terminal and not request.truncated and
             request.action_required is (not terminal), 'wire termination/truncation mismatch')
    _require(math.isfinite(now) and 0 <= now <= 28800 and
             ((terminal and now == 28800) or (not terminal and now < 28800)), 'false shift terminal/time')
    _require(elapsed >= 0, 'simulation time regressed')
    _close(request.sim_time_delta_seconds, elapsed, 'wire simulation delta', absolute=1e-6)


def replay_recording(directory, *, event_bundles=None, require_events=False, allow_synthetic=False,
                     identity_reconciliation=None):
    """Verify a closed recording and return derived ledgers/MC labels read-only.

    A successful replay confirms consistency of recorded evidence. Missing event
    sidecars are reported explicitly, and ``require_events`` rejects that case.
    It never grants approval to collect data, pretrain, or run PPO.
    """
    directory = Path(directory)
    if identity_reconciliation is not None:
        from .offline_identity_reconciliation import validate
        validate(directory, identity_reconciliation)
    _require(not (directory/'ABORTED.json').exists(), 'recording has ABORTED marker; normal completion is not valid')
    manifest = _object(directory/'recording_manifest.json')
    if 'execution_options' in manifest:
        raw=(directory/'execution_options.json').read_bytes()
        _require(_sha(raw)==manifest.get('execution_options_sha256')
                 and json.loads(raw)==manifest['execution_options'],'execution options evidence mismatch')
    _require(manifest.get('schema') == 'boundary-recording-v1', 'unsupported recording schema')
    _require(manifest.get('status') == 'complete', 'recording is incomplete/aborted, not a complete normal shift')
    _require(manifest.get('complete') is True, 'recording complete flag is absent/false')
    scope = manifest.get('scope')
    _require(scope in ('actual_anylogic_acceptance', 'actual_anylogic_offline', 'synthetic_unit_test'), 'unknown recording scope')
    _require(manifest.get('source_kind') == scope, 'recording source/scope mismatch')
    _require(allow_synthetic or scope != 'synthetic_unit_test', 'synthetic recording requires explicit allow_synthetic')
    try:
        cfg_bytes = (directory/'acceptance_run_config.json').read_bytes()
        cfg = json.loads(cfg_bytes)
    except (OSError, ValueError) as exc:
        raise RecordingReplayError(f'cannot read recorded run configuration: {exc}') from exc
    _require(type(cfg) is dict and _sha(cfg_bytes) == manifest.get('config_sha256'),
             'run configuration hash mismatch')
    recording_id = cfg.get('recording', {}).get('recording_id')
    _require(isinstance(recording_id, str) and recording_id and
             recording_id == manifest.get('recording_id'), 'recording identity mismatch')
    _require(manifest.get('observation_dtype') == 'float32' and manifest.get('observation_width') == 1519
             and manifest.get('field_names_sha256') == _sha(('\n'.join(FIELD_NAMES)+'\n').encode()),
             'recorded observation schema metadata mismatch')
    jobs = manifest.get('expected_jobs')
    _require(type(jobs) is list and jobs and all(type(j) is dict for j in jobs), 'missing complete expected jobs')
    try:
        validate_scenario_jobs(cfg,jobs)
    except ValueError as exc:
        raise RecordingReplayError(str(exc)) from exc
    _require(len({j.get('run_index') for j in jobs}) == len(jobs) and
             len({j.get('experiment_seed') for j in jobs}) == len(jobs), 'duplicate job run index/seed')
    wire, responses, unique_order = _wire_stream(_lines(directory, manifest, 'rpc.jsonl'))
    rows = _lines(directory, manifest, 'transitions.jsonl')
    episodes, consumed, seen_ids, pending, current = [], [], set(), None, None
    baseline = None
    for row in rows:
        event = row.get('event')
        _require(event in ('episode_start', 'action', 'transition', 'episode_end', 'episode_aborted'),
                 'unknown transition recording event')
        _require(event != 'episode_aborted', 'recording contains aborted episode')
        episode = row.get('episode_id')
        if event == 'episode_start':
            _require(current is None and len(episodes) < len(jobs) and episode not in seen_ids,
                     'episode overlap/duplicate/unexpected start')
            job = jobs[len(episodes)]
            _require(row.get('job') == job, 'episode seed/scenario job differs from manifest')
            i = row.get('info', {})
            request = wire.get(i.get('request_id'))
            _require(request is not None and request.episode_id == episode, 'missing original start request')
            _info(i, request, 'start info')
            _request_lifecycle(request, sequence=0, now=request.sim_time_seconds, elapsed=0., terminal=False)
            _require(request.sim_time_seconds < 7200 and request.previous_action == -1 and
                     not request.closed_windows and request.transition_reward == 0 and
                     request.reward_components.get('wait_vehicle_seconds') == 0 and
                     request.reward_components.get('delta_wait_vehicle_seconds') == 0,
                     'invalid/unaccounted episode prefix')
            _require((request.experiment_run_index, request.experiment_seed) ==
                     (job.get('run_index'), job.get('experiment_seed')), 'wire seed/job mismatch')
            baseline = _runtime(request, cfg, baseline)
            try:
                validate_reward_accounting(request.transition_reward, dict(request.reward_components), [],
                                           config=baseline, terminal=False)
            except (ValueError, TypeError, KeyError) as exc:
                raise RecordingReplayError(f'invalid start reward: {exc}') from exc
            _observation(row.get('observation'), request, 'start observation')
            seen_ids.add(episode); consumed.append(request.request_id)
            current = dict(episode_id=episode, job=job, request=request, info=i,
                observation=row['observation'], rewards=[], valid=[], windows=[], closed=0,
                cumulative_wait=0., terminal=False, request_ids=[request.request_id],
                masks=[], coverage=dict(empty_window=False, multi_window_crossing=False,
                    last_window_with_shift_end=False), wait_deltas=[])
        elif event == 'action':
            _require(current is not None and current['episode_id'] == episode and not current['terminal']
                     and pending is None, 'action outside active decision or duplicate action')
            request = current['request']
            _require(row.get('request_id') == request.request_id and
                     row.get('decision_sequence') == request.decision_sequence, 'action request correlation mismatch')
            _observation(row.get('observation'), request, 'action observation')
            mask = [bool(v) for v in request.obs.values[19:23]]
            action = row.get('action')
            _require(row.get('action_mask') == mask and type(action) is int and 0 <= action < 4 and mask[action],
                     'illegal/mismatched recorded action or mask')
            _require(responses[request.request_id] == (True, action), 'recorded action differs from wire response')
            pending = row
        elif event == 'transition':
            _require(current is not None and current['episode_id'] == episode and pending is not None
                     and not current['terminal'], 'transition without corresponding action')
            _require(type(row.get('step_index')) is int and row['step_index'] == len(current['rewards']),
                     'noncontiguous transition step index')
            _require(not row.get('truncated') and row.get('truncated') is False and
                     row.get('abnormal_interruption') is False and row.get('sampler_truncated') is False,
                     'truncation/abnormal interruption is not a complete shift')
            terminal = row.get('terminated')
            _require(type(terminal) is bool, 'missing transition termination marker')
            previous = current['request']; i = row.get('next_info', {})
            request = wire.get(i.get('request_id'))
            _require(request is not None and request.episode_id == episode, 'missing original transition request')
            _info(row.get('action_info'), previous, 'action info')
            _info(i, request, 'next info')
            _require(row.get('action') == pending['action'] and row.get('action_mask') == pending['action_mask'],
                     'transition action differs from sent action')
            _observation(row.get('observation'), previous, 'transition observation')
            _require(row['observation'] == current['observation'] == pending['observation'],
                     'discontinuous decision observations')
            _observation(row.get('next_observation'), request, 'next observation')
            _request_lifecycle(request, sequence=len(current['rewards'])+1,
                now=request.sim_time_seconds, elapsed=request.sim_time_seconds-previous.sim_time_seconds,
                terminal=terminal)
            _require(request.previous_action == row['action'], 'wire previous_action differs from applied action')
            _require((request.experiment_run_index, request.experiment_seed) ==
                     (previous.experiment_run_index, previous.experiment_seed), 'episode provenance changed')
            runtime = _runtime(request, cfg, baseline)
            c = dict(request.reward_components)
            cumulative, delta = c.get('wait_vehicle_seconds'), c.get('delta_wait_vehicle_seconds')
            _require(type(cumulative) in (int, float) and math.isfinite(cumulative) and
                     cumulative >= current['cumulative_wait'], 'cumulative waiting regressed')
            _close(delta, cumulative-current['cumulative_wait'], 'independent waiting increment', absolute=1e-7)
            before = current['closed']; valid = [j >= before for j in range(4)]
            costs = [0.]*4
            settlements = [window_summary_to_dict(w) for w in request.closed_windows]
            for w in settlements:
                j = current['closed']
                _require(j < 4 and w['window_index'] == j and w['window_end_seconds'] <= request.sim_time_seconds,
                         'missing/duplicate/future/out-of-order window settlement')
                try:
                    costs[j] = validate_window_summary(w, config=runtime)
                except (ValueError, TypeError, KeyError) as exc:
                    raise RecordingReplayError(f'invalid window settlement: {exc}') from exc
                current['windows'].append(w); current['closed'] += 1
            after = current['closed']
            _require(after == min(4, int(request.sim_time_seconds//7200)), 'crossed window settlement missing')
            bootstrap = [j >= after and valid[j] and not terminal for j in range(4)]
            expected = dict(grade_valid=valid, grade_costs=costs, grade_bootstrap=bootstrap,
                grade_trace=bootstrap, next_grade_valid=[j >= after for j in range(4)],
                reward_bootstrap=not terminal, reward_trace=not terminal)
            for name, value in expected.items():
                _require(row.get(name) == value, f'independent {name} boundary mismatch')
            # Recompute the reward from actual wait/production quantities, then
            # check all representations. The adapter result is not an oracle.
            formal = -runtime.get('wait_kappa')*delta
            if terminal:
                formal -= runtime.get('w_production')*max(0., 1.-c['ore_tons_shift']/16000.)
                formal -= runtime.get('waste_shortfall_weight')*max(0., 1.-c['waste_tons_shift']/4000.)
            try:
                validate_reward_accounting(request.transition_reward, c, settlements, config=runtime, terminal=terminal)
            except (ValueError, TypeError, KeyError) as exc:
                raise RecordingReplayError(f'invalid production reward: {exc}') from exc
            for name in ('env_reward', 'raw_reward', 'formal_reward'):
                _close(row.get(name), formal, f'{name} independent reward')
            _close(request.transition_reward, formal, 'wire independent reward')
            current['rewards'].append(formal); current['valid'].append(valid)
            current['masks'].append(expected); current['wait_deltas'].append(delta)
            current['coverage']['empty_window'] |= any(w['ore_tons'] == 0 for w in settlements)
            current['coverage']['multi_window_crossing'] |= len(settlements) > 1
            current['coverage']['last_window_with_shift_end'] |= terminal and any(w['window_index'] == 3 for w in settlements)
            current.update(request=request, info=i, observation=row['next_observation'],
                           cumulative_wait=cumulative, terminal=terminal)
            consumed.append(request.request_id); current['request_ids'].append(request.request_id)
            pending = None
        else:  # episode_end
            _require(current is not None and current['episode_id'] == episode and current['terminal']
                     and pending is None and current['closed'] == 4, 'premature/incomplete episode end')
            _require(row.get('job') == current['job'], 'episode end job mismatch')
            _info(row.get('final_info'), current['request'], 'final info')
            _require(row.get('transitions') == len(current['rewards']), 'episode transition count mismatch')
            rewards = current['rewards']; total = math.fsum(rewards)
            _close(row.get('total_reward'), total, 'episode total reward', absolute=1e-8)
            terminal_c = dict(current['request'].reward_components)
            ledger = -baseline.get('wait_kappa')*current['cumulative_wait']
            ledger -= baseline.get('w_production')*max(0., 1.-terminal_c['ore_tons_shift']/16000.)
            ledger -= baseline.get('waste_shortfall_weight')*max(0., 1.-terminal_c['waste_tons_shift']/4000.)
            _close(total, ledger, 'full-shift independent ledger', absolute=1e-8)
            costs = [float(w['violation_lower'] or w['violation_upper']) for w in current['windows']]
            carry, mc_reward = 0., [0.]*len(rewards)
            for t in range(len(rewards)-1, -1, -1):
                carry += rewards[t]; mc_reward[t] = carry
            mc_grade = [[cost if active else 0. for cost, active in zip(costs, mask)]
                        for mask in current['valid']]
            episodes.append(dict(episode_id=episode, job=current['job'], transitions=len(rewards),
                total_reward=total, wait_vehicle_seconds=current['cumulative_wait'],
                wait_deltas=current['wait_deltas'], windows=current['windows'], window_costs=costs,
                masks=current['masks'], mc_reward=mc_reward, mc_grade=mc_grade,
                request_ids=current['request_ids'], terminal_components=terminal_c,
                material_residuals=_material(terminal_c, current['windows'], baseline),
                coverage=current['coverage']))
            current = None
    _require(current is None and pending is None and len(episodes) == len(jobs), 'missing complete episode end/jobs')
    _require(consumed == unique_order, 'missing/unconsumed/reordered original requests')
    entries = manifest.get('episodes')
    _require(type(entries) is list and len(entries) == len(episodes), 'manifest episode count mismatch')
    for saved, derived in zip(entries, episodes):
        _require(saved.get('episode_id') == derived['episode_id'], 'manifest episode identity mismatch')
        _require(saved.get('normal_completion') is True, 'manifest episode lacks normal completion')
        for key in ('job', 'transitions'):
            if key in saved:
                _require(saved[key] == derived[key], f'manifest episode {key} mismatch')
        _close(saved.get('total_reward'), derived['total_reward'], 'manifest total reward', absolute=1e-8)

    events, event_ids = [], set()
    if event_bundles:
        from .acceptance_events import replay_event_bundle
        for bundle in event_bundles:
            expected_recording_id = recording_id
            if identity_reconciliation is not None:
                from .offline_identity_reconciliation import event_recording_id
                expected_recording_id = event_recording_id(directory, identity_reconciliation, bundle)
            evidence = replay_event_bundle(bundle,
                expected_config_sha256=manifest['config_sha256'],
                expected_recording_id=expected_recording_id,
                expected_source_kind='synthetic_unit_test' if scope == 'synthetic_unit_test' else 'actual_anylogic')
            identity = evidence['episode_id']
            _require(identity not in event_ids and identity in seen_ids, 'duplicate/unexpected event episode')
            event_ids.add(identity)
            episode = next(e for e in episodes if e['episode_id'] == identity)
            _join_events(evidence, episode, wire, responses)
            events.append(evidence)
    event_verified = event_ids == seen_ids
    _require(not require_events or event_verified, 'independent waiting/FIFO event evidence is missing')
    return dict(schema='boundary-recording-replay-v1', scope=scope,
        status='recording_verified_requires_review', recording_id=recording_id,
        config_sha256=manifest['config_sha256'], training_executed=False,
        trainable_dataset=False, stage1_acceptance_passed=False,
        actual_observation_wire_correspondence_verified=True,
        future_information_absence_verified=False,
        event_evidence_verified=event_verified, missing_event_episodes=sorted(seen_ids-event_ids),
        identity_reconciliation_used=identity_reconciliation is not None,
        strict_identity_passed=event_verified and identity_reconciliation is None,
        requests=len(wire), actions=sum(r[0] for r in responses.values()),
        transitions=sum(e['transitions'] for e in episodes), episodes=episodes, events=events,
        limitations=[
            'Wire-to-input equality and schema hashes do not prove upstream observations contain no future information.',
            'Without event evidence, wait/FIFO checks reconcile RPC summaries only and do not independently reconstruct physical events.',
            'Successful replay verifies recorded consistency; it does not certify scenario coverage or authorize training.'])


def _join_events(evidence, episode, wire, responses):
    job = episode['job']
    _require(evidence['seed'] == job['experiment_seed'] and evidence['run_index'] == job['run_index'],
             'event/wire seed mismatch')
    terminal = episode['terminal_components']
    for key in ('shift_seconds', 'window_seconds', 'grade_lower', 'grade_upper',
                'grade_epsilon', 'tonnage_epsilon', 'mass_balance_epsilon'):
        _require(evidence['config'].get(key) == terminal.get('config.'+key),
                 'event/wire configuration mismatch: '+key)
    # Capacity is an actual 1519D input even though the RPC configuration map
    # does not repeat it. Compare in float32, matching the wire's exact scale.
    for identity in episode['request_ids']:
        values = wire[identity].obs.values
        crusher_seen = False
        for site in range(2):
            if values[FIELD_NAMES.index(f'dump.{site}.acceptsOre')] == 1.:
                crusher_seen = True
                _require(values[FIELD_NAMES.index(f'dump.{site}.crusherRate')] ==
                         float(np.float32(evidence['config']['crusher_rate_tph']/2500.)),
                         'event/wire crusher physical rate mismatch')
        _require(crusher_seen, 'no ore crusher identified in actual observation')
    requests = evidence.get('requests', evidence.get('decisions'))
    _require([r['request_id'] for r in requests] == episode['request_ids'], 'event/wire request coverage mismatch')
    for event in requests:
        request = wire[event['request_id']]
        _require(event['decision_sequence'] == request.decision_sequence and
                 event['terminal'] == (request.kind == pb.EPISODE_END), 'event/wire lifecycle mismatch')
        _close(event['sim_time_seconds'], request.sim_time_seconds, 'event/wire request time', absolute=1e-6)
        _close(event['cumulative_wait_seconds'], request.reward_components['wait_vehicle_seconds'],
               'independent event/wire cumulative waiting', absolute=1e-7)
        _close(event['delta_wait_seconds'], request.reward_components['delta_wait_vehicle_seconds'],
               'independent event/wire delta waiting', absolute=1e-7)
        _require(event['window_indices'] == list(range(min(4, int(request.sim_time_seconds//7200)))),
                 'event/wire closed window coverage mismatch')
        if request.action_required:
            _require(responses[request.request_id] == (True, event['action']), 'event/wire applied action mismatch')
    for event, window in zip(evidence['windows'], episode['windows']):
        _require(event['window_index'] == window['window_index'] and
                 event['cost'] == int(window['violation_lower'] or window['violation_upper']),
                 'independent event/wire window cost mismatch')
        for field in ('ore_tons', 'metal_tons'):
            _close(event[field], window[field], 'independent event/wire '+field,
                   absolute=evidence['config']['mass_balance_epsilon'])
    c = episode['terminal_components']; eps = evidence['config']['mass_balance_epsilon']
    for actual, key in ((evidence['ore_tons'], 'ore_tons_shift'), (evidence['waste_tons'], 'waste_tons_shift'),
                        (sum(b['tons'] for b in evidence['fifo_remaining']), 'crusher_buffer_tons'),
                        (evidence['material_totals']['enqueued_tons'], 'crusher_received_tons'),
                        (sum(p['tons'] for p in evidence['in_transit_payloads'].values()
                             if p['material'] == 'ore'), 'carried_ore_tons'),
                        (sum(p['tons'] for p in evidence['in_transit_payloads'].values()
                             if p['material'] == 'waste'), 'carried_waste_tons')):
        _close(actual, c[key], 'independent event/wire terminal '+key, absolute=eps)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify',))
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--events', type=Path, action='append')
    parser.add_argument('--require-events', action='store_true')
    parser.add_argument('--allow-synthetic', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        result = replay_recording(args.recording, event_bundles=args.events,
            require_events=args.require_events, allow_synthetic=args.allow_synthetic)
        output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
        if args.output is None:
            print(output, end='')
        else:
            with args.output.open('x', encoding='utf-8') as target:
                target.write(output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('REPLAY REFUSED: '+str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
