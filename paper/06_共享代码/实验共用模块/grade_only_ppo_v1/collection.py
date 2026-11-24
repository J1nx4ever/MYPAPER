"""Fixed-Actor RPC capture and acknowledged compact-journal admission.

The inherited collector acknowledges the true terminal callback before this
wrapper waits for Java's final journal. No simulator is launched here.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import time

import numpy as np

from baseline1_v1.collection import OnlineActor, PhaseCollector, verify_durable_reference
from grade_event_memory_v1.admission import (
    _check_query_physics, _phase, admit_episode, frozen_actor, reference, require,
)
from grade_event_memory_v1.events import build_memory
from grade_event_memory_v1.journal import read_journal
from grade_event_memory_v1.storage import read_episode_hdf5
from grade_event_memory_v1.targets import REWARD_VERSION


MATERIAL_RPC_FIELDS = {
    'config.material_event_memory_enabled': 1.0,
    'config.material_event_contract_version': 1.0,
}
RECEIPT_SCHEMA = 'grade-only-material-collection-receipt-v1'


def material_journal_path(job):
    """One new, explicit job-local identity. Old acceptance JSONL stays off."""
    directory = Path(job['material_event_bundle'])
    require(directory.is_absolute() and directory == directory.resolve(),
            'material event directory must be absolute and normalized')
    if 'material_event_directory' in job:
        require(directory.parent == Path(job['material_event_directory']),
                'material episode bundle must be an immediate child of configured event root')
    return directory / 'material_events.bin'


def wait_for_material_journal(cfg, job, *, clock=None, sleeper=None):
    """Call only after inherited collect returns (the terminal RPC is ACKed)."""
    clock = time.monotonic if clock is None else clock
    sleeper = time.sleep if sleeper is None else sleeper
    timeout = cfg['execution']['event_finalize_timeout_seconds']
    require(type(timeout) in (int, float) and np.isfinite(timeout) and timeout > 0,
            'explicit positive material finalize timeout required')
    deadline = clock() + timeout
    path = material_journal_path(job)
    while not path.exists():
        require(clock() < deadline,
                'missing acknowledged compact material END; preserve raw data, never resubmit')
        sleeper(min(.05, max(0., deadline-clock())))
    # A final filename is an atomic publication; malformed finals are rejected
    # immediately rather than treated as a file that might eventually improve.
    journal = read_journal(path)
    require(journal.header['source_kind'] == 'actual_anylogic',
            'formal RPC collection requires actual AnyLogic material evidence')
    return path


def _references(config_path, actor_path, journal_path, raw_manifest, admitted_path):
    return dict(config=reference(config_path), actor=reference(actor_path),
                journal=reference(journal_path), raw_manifest=reference(raw_manifest),
                episode=reference(admitted_path),
                episode_manifest=reference(str(admitted_path)+'.manifest.json'),
                episode_seal=reference(str(admitted_path)+'.seal.json'))


def _same_references(refs):
    require(set(refs) == {'config','actor','journal','raw_manifest','episode',
                          'episode_manifest','episode_seal'}, 'incomplete material source bindings')
    for ref in refs.values():
        require(reference(ref['path']) == ref, 'material source bytes changed after admission')
    verify_durable_reference(refs['raw_manifest'])


def load_admitted_episode(receipt, *, allow_synthetic=False):
    """Rebind a saved episode to raw RPC, config, Actor and journal bytes.

    This is deliberately a substantive validator, not a permissive callback to
    bypass the nonsynthetic HDF5 gate. Production shortfalls remain in the data.
    The caller must pin the receipt itself in its controller/checkpoint chain.
    """
    import torch
    from python.collect_offline import validate_runtime
    from python.constrained_batch import load_dataset
    from python.reward_config import configuration_from_components
    from python.runtime_scenario import PREFIX, fingerprint

    require(receipt.get('schema') == RECEIPT_SCHEMA and receipt.get('normal_completion') is True,
            'complete material receipt required')
    refs = receipt['references']
    _same_references(refs)
    require('phase_config' not in receipt or receipt['phase_config']==refs['config'],
            'completion receipt phase binding differs from admitted configuration')
    cfg = _phase(refs['config']['path'])
    journal = read_journal(refs['journal']['path'])
    header = journal.header
    require(all(name not in receipt or receipt[name]==header[field] for name,field in
                (('run_index','run_index'),('experiment_seed','seed'))),
            'completion receipt run/seed differs from material journal')
    synthetic = header['source_kind'] == 'synthetic'
    require(not synthetic or allow_synthetic, 'synthetic material requires explicit opt-in')
    require(header['source_kind'] in ('actual_anylogic','synthetic'), 'unsupported material source')
    require(header['config_sha256'] == refs['config']['sha256'] and
            header['recording_id'] == cfg['recording_id'], 'material configuration binding differs')
    actor, policy = frozen_actor(refs['actor']['path'], allow_small_synthetic=synthetic)
    jobs = [job for job in cfg['jobs'] if job['run_index'] == header['run_index']]
    require(len(jobs) == 1, 'material run must match exactly one configured job')
    job = jobs[0]
    require(job['experiment_seed'] == header['seed'], 'material environment seed differs')
    raw_dir = Path(refs['raw_manifest']['path']).parent
    rows, episodes, _, _, manifest = load_dataset(raw_dir, require_real=not synthetic)
    try:
        require(manifest['source_kind'] == ('synthetic_unit_test' if synthetic else 'real_anylogic_rpc'),
                'raw and journal source kinds differ')
        require(len(episodes) == 1, 'material receipt must bind one complete shift')
        raw = episodes[0]
        require(raw['episode_id'] == header['episode_id'] and
                raw['experiment_run_index'] == header['run_index'] and
                raw['experiment_seed'] == header['seed'] and raw['policy_version'] == policy,
                'raw RPC/material/Actor identity differs')
        infos = [raw['start_info']] + [row['settlement_info'] for row in rows]
        expected = job['expected_rpc_config']
        for info in infos:
            actual = configuration_from_components(info['reward_components'], terminal=info is infos[-1])
            require(all(actual.to_dict().get(k) == v for k,v in expected.items()),
                    'RPC runtime configuration differs from frozen phase')
            require(actual.get('target_grade') == header['target_grade'] and
                    actual.get('grade_lower') == header['grade_lower'] and
                    actual.get('grade_upper') == header['grade_upper'], 'material grade bounds differ')
            if not synthetic:
                validate_runtime(info, job)
        if not synthetic:
            require(all(expected.get(k) == v for k,v in MATERIAL_RPC_FIELDS.items()),
                    'RPC must explicitly enable the compact material contract')
            require(job['partition'] in ('train','validation') and
                    job['runtime_scene_sha256'] == fingerprint({k:v for k,v in expected.items()
                                                               if k.startswith(PREFIX)}),
                    'frozen split/scene fingerprint differs')
            require(manifest.get('provenance',{}).get('configuration') == refs['config'],
                    'raw RPC is not bound to these phase bytes')
        memory = build_memory(journal.events)
        markers = [event for event in journal.events if event.kind in ('DECISION','END')]
        require(len(markers) == len(infos) and len(memory.queries) == len(rows),
                'RPC/material marker count differs')
        for marker, info in zip(markers, infos):
            require((marker.request_id,marker.decision_sequence,marker.time) ==
                    (info['request_id'],info['decision_sequence'],info['sim_time_seconds']),
                    'RPC/material causal marker differs')
        for settlement, window in zip(memory.settlements, raw['windows']):
            require(all(np.isclose(value,window[key],rtol=1e-10,atol=1e-8) for value,key in
                        ((settlement.tons,'ore_tons'),(settlement.metal,'metal_tons'),
                         (settlement.grade_reward_raw,'grade_reward_raw'))),
                    'RPC/material window settlement differs')
        states = np.stack([rows[0]['observation']] + [row['next_observation'] for row in rows]).astype(np.float32)
        actions = np.asarray([row['action'] for row in rows],dtype=np.int64)
        log_probs = np.asarray([row['old_log_prob'] for row in rows],dtype=np.float64)
        require(all(row['policy_version'] == policy and not row.get('critic_only',False) and
                    row.get('behavior_kind') != 'anylogic_native' for row in rows),
                'raw rows are not fixed Actor on-policy trajectories')
        _check_query_physics(journal.events, states)
        with torch.no_grad():
            sampled = actor.distribution(torch.tensor(states[:-1])).log_prob(torch.tensor(actions)).numpy()
        require(np.allclose(sampled,log_probs,rtol=1e-6,atol=1e-6), 'stored Actor probabilities differ')
        identity = receipt['identity']
        bound = dict(recording_id=header['recording_id'], episode_id=header['episode_id'],
                     run_index=header['run_index'], environment_seed=header['seed'],
                     actor_sampling_seed=job['actor_sampling_seed'], policy_version=policy,
                     policy_sha256=refs['actor']['sha256'], config_sha256=refs['config']['sha256'],
                     journal_file_sha256=refs['journal']['sha256'],
                     journal_logical_sha256=journal.logical_sha256,
                     raw_manifest_sha256=refs['raw_manifest']['sha256'],
                     raw_process_calls=journal.raw_process_calls,
                     tokenization_version=header['tokenization_version'],
                     observation_schema='dispatch.physical.v3',reward_version=REWARD_VERSION,
                     source_kind='synthetic' if synthetic else 'fixed_actor',
                     split='synthetic' if synthetic else job['partition'])
        require(all(identity.get(k) == v for k,v in bound.items()), 'admitted source identity differs')
        require(identity.get('scene_id') == raw['scene_id'] and
                identity.get('scene_group') == raw['scene_group'], 'admitted scene identity differs')
        if not synthetic:
            require(job['scene_id'] == raw['scene_id'] and job['scene_group'] == raw['scene_group'],
                    'configured scene identity differs from RPC')
        terminal = infos[-1]['reward_components']
        def validate(derived):
            _same_references(refs)
            require(derived['identity'] == identity and derived['memory'] == memory,
                    'derived identity or causal material tape differs')
            require(np.array_equal(derived['states'],states) and
                    np.array_equal(derived['actions'],actions) and
                    np.array_equal(derived['old_log_probs'],log_probs),
                    'derived state/action/probability differs from sealed RPC')
            require(all(derived['evaluation'][target] == float(terminal[source]) for target,source in
                        (('ore_tons','ore_tons_shift'),('waste_tons','waste_tons_shift'),
                         ('total_wait_seconds','wait_vehicle_seconds'))),
                    'derived production/waiting ledger differs from RPC')
        return read_episode_hdf5(refs['episode']['path'], expected_identity=identity, validator=validate)
    finally:
        if hasattr(rows,'close'): rows.close()


class MaterialPhaseCollector:
    """One immutable phase Actor, compact journal mandatory, full JSONL disabled."""
    def __init__(self, cfg, config_path, *, publish_ready, startup=None):
        self.cfg, self.config_path = copy.deepcopy(cfg), Path(config_path).resolve()
        frozen_cfg=json.loads(self.config_path.read_bytes())
        require({k:v for k,v in self.cfg.items() if k!='jobs'} ==
                {k:v for k,v in frozen_cfg.items() if k!='jobs'}, 'phase configuration bytes differ')
        require(bool(self.cfg.get('jobs')) and self.cfg['jobs'] ==
                [job for job in frozen_cfg['jobs'] if job in self.cfg['jobs']],
                'resumed sampling jobs must preserve their exact frozen phase rows and order')
        require(self.cfg.get('capture_physical_events') is False, 'legacy full physical event capture must be off')
        _phase(self.config_path)
        paths = [material_journal_path(job) for job in self.cfg['jobs']]
        require(len(set(paths)) == len(paths), 'duplicate material event identity')
        for job in self.cfg['jobs']:
            require(all(job['expected_rpc_config'].get(k) == v for k,v in MATERIAL_RPC_FIELDS.items()),
                    'RPC must explicitly enable compact material recording')
        self._collector = PhaseCollector(self.cfg,self.config_path,publish_ready=publish_ready,startup=startup)
        self._actor_reference = None

    def __enter__(self):
        self._collector.__enter__()
        return self

    def collect(self, job, behavior, *, actor_path):
        require(job in self.cfg['jobs'], 'job absent from immutable phase')
        require(not material_journal_path(job).parent.exists(), 'material identity already used; never recapture')
        actor_reference = reference(actor_path)
        if self._actor_reference is None: self._actor_reference = actor_reference
        require(self._actor_reference == actor_reference, 'Actor snapshot changed inside one sampling phase')
        frozen, policy = frozen_actor(actor_path)
        behavior.assert_frozen()
        require(behavior.policy_version == policy, 'sampling behavior differs from pinned Actor')
        # Java exclusively creates the episode child at Main startup. Only its
        # parent may be created here, before inherited publish_ready is called.
        material_journal_path(job).parent.parent.mkdir(parents=True,exist_ok=True)
        # This inherited call completes and ACKs terminal RPC before returning.
        raw, raw_ref, original = self._collector.collect(job,behavior)
        try:
            journal = wait_for_material_journal(self.cfg,job)
            output = Path(self.cfg['python_output'])/'material'/f"run_{job['run_index']}.h5"
            admission = admit_episode(Path(raw_ref['path']).parent,journal,
                config_path=self.config_path,actor_path=actor_path,output_path=output,split=job['partition'])
            require(reference(actor_path) == actor_reference, 'Actor snapshot changed while collecting')
            behavior.assert_frozen()
            receipt = dict(schema=RECEIPT_SCHEMA,normal_completion=True,identity=admission['identity'],
                           phase_id=self.cfg['phase_id'],role=job['role'],raw_receipt=original,
                           run_index=job['run_index'],experiment_seed=job['experiment_seed'],
                           phase_config=reference(self.config_path),
                           references=_references(self.config_path,actor_path,journal,raw_ref['path'],output),
                           four_windows=4,transitions=admission['transitions'],
                           training_objective='negative_grade_cost_only',material_evidence=True,
                           production_guarantee='pending',production_failures_retained=True)
            episode = load_admitted_episode(receipt)
            path = Path(raw_ref['path']).parent/'material_receipt.json'
            with path.open('xb') as stream:
                stream.write((json.dumps(receipt,sort_keys=True,allow_nan=False)+'\n').encode())
                stream.flush(); os.fsync(stream.fileno())
            return episode, reference(str(output)+'.manifest.json'), receipt
        finally:
            if hasattr(raw.rows,'close'): raw.rows.close()

    def __exit__(self,*error):
        return self._collector.__exit__(*error)
