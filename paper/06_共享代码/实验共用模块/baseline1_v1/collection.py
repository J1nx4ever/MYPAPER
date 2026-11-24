"""Per-phase RPC capture with durable complete-shift inputs and no PPO update."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

from adaptation_v1_1.collect import FrozenJobActor
from python.collect_offline import validate_runtime
from python.constrained_batch import EpisodeBatchBuffer, TrainingBatch, collect_batch, load_dataset, validate_rows
from python.native_behavior import NativeBehavior
from python.storage_contract import CompositeRows, HDF5, storage_options
from .gate import encoded, reference, require, sha
from .prepared_batch import PreparedRows, prepare_rows, join_prepared_rows, make_raw_batch


class OnlineActor(FrozenJobActor):
    def row_metadata(self, info):
        result = super().row_metadata(info)
        result['critic_only'] = False
        return result


def join_buffers(buffers):
    require(bool(buffers), 'empty complete batch')
    started = time.perf_counter()
    if all(isinstance(b.rows, PreparedRows) and b.episodes is b.rows.episodes for b in buffers):
        source_kind = buffers[0].source_kind
        episodes, offset = [], 0
        for b in buffers:
            require(b.complete and b.source_kind == source_kind, 'incomplete/mixed-source batch')
            for episode in b.episodes:
                e = dict(episode)
                e.update(start=e['start'] + offset, stop=e['stop'] + offset)
                episodes.append(e)
            offset += len(b.rows)
        raw = make_raw_batch(join_prepared_rows([b.rows for b in buffers], episodes), source_kind)
        raw.preparation_timing = dict(getattr(buffers, 'preparation_timing', {}),
            join_seconds=time.perf_counter() - started, immutable_components_reused=len(buffers))
        return raw
    target = EpisodeBatchBuffer(expected_episodes=sum(len(b.episodes) for b in buffers),
                                source_kind=buffers[0].source_kind)
    parts = []
    offset = 0
    for b in buffers:
        require(b.complete and b.source_kind == target.source_kind, 'incomplete/mixed-source batch')
        parts.append(copy.deepcopy(b.rows))
        for episode in b.episodes:
            e = copy.deepcopy(episode)
            e.update(start=e['start'] + offset, stop=e['stop'] + offset)
            target.episodes.append(e)
        offset += len(b.rows)
    target.rows = CompositeRows(parts)
    target.seal()
    raw = TrainingBatch.from_buffer(target)
    raw.source_kind = target.source_kind
    raw.preparation_timing = dict(getattr(buffers, 'preparation_timing', {}),
        join_seconds=time.perf_counter() - started, immutable_components_reused=0)
    return raw


def replay_physical(cfg, job, buffer, config_sha256):
    """Scoped native calibration evidence; production training keeps events off."""
    from python.acceptance_events import replay_event_bundle
    episode = buffer.episodes[0]
    infos = [episode['start_info']] + [r['settlement_info'] for r in buffer.rows]
    checks = []
    for index, info in enumerate(infos):
        c = info['reward_components']
        checks.append(dict(request_id=info['request_id'], decision_sequence=info['decision_sequence'],
            sim_time_seconds=info['sim_time_seconds'], action=buffer.rows[index]['action'] if index < len(buffer.rows) else None,
            cumulative_wait_seconds=c['wait_vehicle_seconds'], delta_wait_seconds=c['delta_wait_vehicle_seconds']))
    audit = replay_event_bundle(job['event_bundle'], expected_episode_id=episode['episode_id'],
        expected_config_sha256=config_sha256, expected_source_kind='actual_anylogic',
        expected_recording_id=cfg['recording_id'], transition_checks=checks)
    require(audit['run_index'] == job['run_index'] and audit['seed'] == job['experiment_seed'],
            'independent physical identity differs')
    for observed, expected in zip(audit['windows'], episode['windows']):
        require(observed['window_index'] == expected['window_index'], 'independent window order differs')
        for key in ('ore_tons', 'metal_tons'):
            require(abs(observed[key] - expected[key]) <= cfg['reward']['mass_balance_epsilon'],
                    'independent window material differs')
    terminal = buffer.rows[-1]['settlement_info']['reward_components']
    require(math.isclose(audit['independent_wait_seconds'], terminal['wait_vehicle_seconds'],
                         rel_tol=1e-10, abs_tol=1e-7), 'independent waiting differs')
    for k, field in (('ore_tons', 'ore_tons_shift'), ('waste_tons', 'waste_tons_shift')):
        require(abs(audit[k] - terminal[field]) <= cfg['reward']['mass_balance_epsilon'],
                'independent terminal production differs')
    return dict(event_manifest=reference(Path(job['event_bundle']) / 'events_manifest.json'),
                events_sha256=audit['events_sha256'], event_count=audit['event_count'],
                independent_wait_seconds=audit['independent_wait_seconds'],
                fifo_replayed=True, coverage=audit['coverage'], limitations=audit['limitations'])


def _wait_event_end(cfg, job):
    deadline = time.monotonic() + cfg['execution']['event_finalize_timeout_seconds']
    path = Path(job['event_bundle']) / 'events_manifest.json'
    while True:
        try:
            data = json.loads(path.read_bytes())
        except (FileNotFoundError, json.JSONDecodeError):
            # The existing Java recorder briefly truncates this file on finish.
            data = None
        if data is not None:
            # Its START manifest deliberately says truncated=true until finish.
            if data.get('complete') is True:
                require(data.get('terminated') is True and data.get('truncated') is False,
                        'physical event END is not normal')
                return
        require(time.monotonic() < deadline, 'missing acknowledged physical EPISODE_END')
        time.sleep(.05)


def load_one(dataset_ref, *, require_real=True):
    started = time.perf_counter()
    require(sha(dataset_ref['path']) == dataset_ref['sha256'], 'durable dataset manifest changed')
    path = Path(dataset_ref['path']).parent
    rows, episodes, _, _, manifest = load_dataset(path, require_real=require_real)
    require(len(episodes) == 1, 'one durable receipt must contain exactly one shift')
    strict_load_seconds = time.perf_counter() - started
    snapshot_started = time.perf_counter()
    native = bool(rows) and (rows[0].get('behavior_kind') == 'anylogic_native'
                            or rows[0].get('critic_only', False))
    if native:
        prepared = None  # Preserve strict native-policy dict compatibility and admission.
    else:
        try:
            prepared = prepare_rows(rows, episodes)
        finally:
            if hasattr(rows, 'close'):
                rows.close()
    buffer = EpisodeBatchBuffer(expected_episodes=1, source_kind=manifest['source_kind'])
    buffer.rows, buffer.episodes, buffer.complete = (rows, episodes, True) if native else (prepared, prepared.episodes, True)
    buffer.preparation_timing = dict(strict_load_seconds=strict_load_seconds,
        immutable_snapshot_seconds=0. if native else time.perf_counter() - snapshot_started)
    buffer.dataset_ref = copy.deepcopy(dataset_ref)
    return buffer


def verify_durable_reference(dataset_ref):
    """Rebind an admitted in-memory snapshot to unchanged durable payload bytes."""
    require(reference(dataset_ref['path']) == dataset_ref, 'durable dataset manifest changed')
    path = Path(dataset_ref['path'])
    manifest = json.loads(path.read_bytes())
    require(manifest.get('status') == 'complete' and not (path.parent / 'ABORTED.json').exists(),
            'durable dataset no longer complete')
    for name, digest in manifest['files_sha256'].items():
        require(sha(path.parent / name) == digest, 'durable dataset payload changed')
    return manifest


def _java_start_properties(path):
    pairs = [line.split('=', 1) for line in Path(path).read_text().splitlines() if line]
    require(all(len(pair) == 2 for pair in pairs), 'malformed Java start claim')
    values = dict(pairs)
    require(len(values) == len(pairs), 'duplicate Java start claim property')
    return values


def wait_for_java_start(startup, job, config_sha256, callback_timeout, *, sleeper=None, clock=None):
    """Wait outside the RPC timer, then verify the existing exclusive Java receipt."""
    sleeper = time.sleep if sleeper is None else sleeper
    clock = time.monotonic if clock is None else clock
    control = Path(startup['control_directory'])
    claim = control / 'java_claims' / f"run_{job['run_index']}.once.claim"
    expected = dict(protocol='baseline1-one-click-java-start-v1', campaign_id=startup['campaign_id'],
        master_plan_sha256=startup['master_plan_sha256'], ordinal=str(job['ordinal']),
        run_index=str(job['run_index']), snapshot_sha256=config_sha256, config_sha256=config_sha256)
    observed_at = None
    print(f"WAITING_FOR_MANUAL_RUN run_index={job['run_index']}; RPC callback timer has not started", flush=True)
    while True:
        require(not (control / 'STOP.properties').exists(), 'preserved STOP while waiting for Java start')
        if claim.exists():
            if observed_at is None:
                observed_at = clock()
            try:
                values = _java_start_properties(claim)
            except ValueError:
                # Java CREATE_NEW precedes writing/fsync; never accept a partial receipt.
                values = {}
            if set(values) >= set(expected):
                require(values == expected, 'exclusive Java start identity differs')
                print(f"JAVA_START_VERIFIED run_index={job['run_index']}; starting existing RPC callback timeout", flush=True)
                return
            require(clock() - observed_at < callback_timeout, 'incomplete Java start claim')
        sleeper(.1)


class PhaseCollector:
    """Open only after caller checks approval; start-ready is published externally."""
    def __init__(self, cfg, config_path, *, publish_ready, startup=None):
        self.cfg, self.config_path, self.publish_ready = copy.deepcopy(cfg), Path(config_path), publish_ready
        self.startup = copy.deepcopy(startup)
        if self.startup is not None:
            require(set(self.startup) == {'control_directory', 'campaign_id', 'master_plan_sha256'},
                    'explicit campaign startup identity required')
            require(Path(self.startup['control_directory']).is_absolute()
                    and self.startup['campaign_id'] == self.cfg['campaign_id']
                    and isinstance(self.startup['master_plan_sha256'], str)
                    and len(self.startup['master_plan_sha256']) == 64
                    and all(c in '0123456789abcdef' for c in self.startup['master_plan_sha256']),
                    'invalid campaign startup identity')
        self.env = self.server = None

    def __enter__(self):
        from python.dispatch_bridge import DispatchCallbackBridge
        from python.dispatch_env import DispatchEnv
        from python.dispatch_service import create_server
        from python.episode_runner import ParameterVariationEpisodeRunner
        storage = storage_options(self.cfg.get('storage'))
        if storage['format'] == HDF5:
            from python.hdf5_storage import _h5py
            _h5py()  # Fail before opening a listener when the dependency is absent.
        p = Path(self.cfg['python_output'])
        p.mkdir(parents=True, exist_ok=True)
        snapshot = p / 'phase_config.json'
        if snapshot.exists():
            require(snapshot.read_bytes() == self.config_path.read_bytes(), 'phase snapshot changed')
        else:
            snapshot.open('xb').write(self.config_path.read_bytes())
        if self.cfg['capture_physical_events']:
            Path(self.cfg['event_directory']).mkdir(parents=True, exist_ok=True)
        jobs = self.cfg['jobs']
        bridge = DispatchCallbackBridge(runtime_configuration=True,
            expected_episode_metadata=[(j['run_index'], j['experiment_seed']) for j in jobs])
        settings = self.cfg['execution']
        self.env = DispatchEnv(bridge, ParameterVariationEpisodeRunner(len(jobs)), settings['callback_timeout_seconds'])
        self.server = create_server(bridge, host=settings['host'], port=settings['port'])
        try:
            self.server.start()
        except BaseException:
            self.env.close()
            self.server.stop(grace=0)
            raise
        print(f"LISTENING {settings['host']}:{settings['port']} phase={self.cfg['phase_id']}; awaiting manual AnyLogic", flush=True)
        return self

    def collect(self, job, behavior):
        require(job in self.cfg['jobs'], 'job absent from immutable phase')
        output = Path(self.cfg['python_output']) / 'episodes' / f"run_{job['run_index']}"
        require(not output.exists(), 'durable job already exists; never recapture')
        if self.cfg['capture_physical_events']:
            require(not Path(job['event_bundle']).exists(), 'physical event identity already used')
        storage = storage_options(self.cfg.get('storage'))
        buffer = EpisodeBatchBuffer(output, expected_episodes=1, source_kind='real_anylogic_rpc',
            storage_format=storage['format'],chunk_rows=storage['chunk_rows'],
            provenance=dict(recording_id=self.cfg['recording_id'],
                            configuration=reference(self.config_path),run_index=job['run_index'],
                            seed=job['experiment_seed']))
        def validate(info, intended):
            validate_runtime(info, intended)
            if self.cfg['capture_physical_events'] and info['decision_sequence'] == 0:
                manifest = json.loads((Path(job['event_bundle']) / 'events_manifest.json').read_bytes())
                expected = dict(complete=False, source_kind='actual_anylogic',
                    recording_id=self.cfg['recording_id'], config_sha256=sha(self.config_path),
                    run_index=job['run_index'], seed=job['experiment_seed'], episode_id=info['episode_id'])
                require(all(manifest.get(k) == v for k, v in expected.items()),
                        'START event identity differs before the first action')
            if self.cfg['kind'] != 'calibration':
                require(not any(k.startswith('config.offline_native.') for k in info['reward_components']),
                        'native-policy metadata cannot enter online PPO')
        self.publish_ready(job, self.config_path)
        try:
            if self.startup is not None:
                wait_for_java_start(self.startup, job, sha(self.config_path),
                                    self.cfg['execution']['callback_timeout_seconds'])
            collect_batch(self.env, behavior, buffer, [job], reward_settings=self.cfg['reward'],
                          runtime_validator=validate)
            require(buffer.complete and len(buffer.episodes) == 1, 'shift not sealed')
            manifest = json.loads((output/'manifest.json').read_bytes())
            for name in (*manifest['files_sha256'], 'manifest.json'):
                with (output / name).open('rb') as stream:
                    os.fsync(stream.fileno())
            evidence = None
            if self.cfg['capture_physical_events']:
                _wait_event_end(self.cfg, job)
                evidence = replay_physical(self.cfg, job, buffer, sha(self.config_path))
            manifest_ref = reference(output / 'manifest.json')
            if hasattr(buffer.rows, 'close'):
                buffer.rows.close()
            reread = load_one(manifest_ref)
            episode = reread.episodes[0]
            require(episode['experiment_run_index'] == job['run_index']
                    and episode['experiment_seed'] == job['experiment_seed'], 'durable input identity differs')
            receipt = dict(normal_completion=True, run_index=job['run_index'],
                experiment_seed=job['experiment_seed'], role=job['role'], phase_id=job['phase_id'],
                episode_id=episode['episode_id'], policy_version=episode['policy_version'],
                phase_config=reference(self.config_path), physical_evidence=evidence,
                storage=storage, dataset_files=manifest['files_sha256'],
                transitions=len(reread.rows), four_windows=4)
            with (output / 'receipt.json').open('xb') as stream:
                stream.write(encoded(receipt))
                stream.flush()
                os.fsync(stream.fileno())
            descriptor = os.open(str(output), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return reread, manifest_ref, receipt
        except BaseException as exc:
            buffer.abort(exc)
            raise

    def __exit__(self, *error):
        try:
            if self.env is not None:
                self.env.close()
        finally:
            if self.server is not None:
                self.server.stop(grace=2).wait()
        self.server = self.env = None
