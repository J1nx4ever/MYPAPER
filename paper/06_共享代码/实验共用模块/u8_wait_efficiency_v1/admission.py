"""Join sealed RPC data and causal material evidence; never collect or fit."""
import hashlib
import json
from pathlib import Path

import numpy as np

from grade_event_memory_v1.events import SCHEMA, build_memory, ContractError
from grade_event_memory_v1.journal import TOKENIZATION, read_journal
from .storage import write_episode_hdf5, read_episode_hdf5
from grade_event_memory_v1.targets import REWARD_VERSION as MATERIAL_REWARD_VERSION
from .targets import REWARD_VERSION, composite_contract, build_composite_targets

from .waiting_diagnostics import (WAIT_DIAGNOSTICS_CONTRACT, require_wait_rpc_configuration,
                                  validate_rpc_wait_breakdown)

TRAINING_REWARD = 'negative_combined_grade_production_wait_cost_v1'


def _training_contract(cfg):
    """Admit exactly the frozen stage coefficient, never infer it from reward."""
    contract=cfg.get('composite_contract')
    require(isinstance(contract,dict) and type(contract.get('wait_weight')) in (int,float) and
            contract['wait_weight'] in (.1,.15) and
            contract == composite_contract(wait_weight=contract['wait_weight']),
            'explicit combined training reward contract required')
    return contract


def require(condition, message):
    if not condition: raise ContractError(message)


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def frozen_actor(path, *, allow_small_synthetic=False):
    """Load only the independently sealed summary-policy Actor, preserving RNG."""
    import torch
    from .learner import ACTOR_SCHEMA, validate_config, _validate_model_state, _load_checkpoint, _hash
    from .models import actor_from_config, digest
    saved = torch.load(path, map_location='cpu', weights_only=False)
    require(isinstance(saved, dict) and saved.get('schema') == ACTOR_SCHEMA,
            'explicit commitment Actor artifact required')
    require(isinstance(saved.get('config'), dict), 'full Actor configuration missing')
    cfg = validate_config(saved['config'])
    require(allow_small_synthetic or cfg['source_kind'] == 'real_anylogic_rpc',
            'formal Actor cannot use a synthetic configuration')
    if not allow_small_synthetic:
        saved, manifest = _load_checkpoint(path)
        from .features import feature_contract
        require(saved.get('schema') == manifest.get('schema') == ACTOR_SCHEMA and
                saved.get('learner_config_sha256') == manifest.get('config_sha256') == _hash(cfg) and
                saved.get('network') == cfg['network'] and
                saved.get('feature_contract') == feature_contract(cfg['feature_config']) and
                all(saved.get(k) == manifest.get(k) == cfg[k]
                    for k in ('candidate_id', 'execution_profile')),
                'sealed Actor artifact configuration differs')
    with torch.random.fork_rng(devices=[]):
        actor = actor_from_config(cfg)
    _validate_model_state(saved['actor'], actor)
    actor.load_state_dict(saved['actor'], strict=True)
    actor.eval()
    for parameter in actor.parameters(): parameter.requires_grad_(False)
    fingerprint = digest(actor)
    require(saved.get('actor_digest') == fingerprint, 'fixed Actor artifact fingerprint differs')
    actor._artifact_config = cfg
    return actor, fingerprint


def _check_actor_phase(actor, cfg):
    config = actor._artifact_config
    from .features import feature_contract
    require(config['candidate_id'] == cfg.get('candidate_id') and
            config['execution_profile'] == cfg.get('execution_profile') and
            all(job.get('actor_seed') == config['seed'] for job in cfg['jobs']) and
            cfg['feature_contract'] == feature_contract(config['feature_config']) and
            cfg.get('training_reward') == TRAINING_REWARD and
            _training_contract(cfg) == config.get('reward_contract'),
            'Actor candidate/profile/initialization differs from phase')


def _verify_sampling(rows, probabilities, job):
    """Verify actual private-RNG actions and the recorded RNG chain, not labels."""
    import torch
    seed = job['actor_sampling_seed']
    generator = torch.Generator(device='cpu'); generator.manual_seed(seed)
    probabilities = torch.as_tensor(probabilities, dtype=torch.float32)
    require(probabilities.shape == (len(rows), 4), 'sampling probabilities shape differs')
    def rng_sha(): return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
    for index, row in enumerate(rows):
        require(row.get('actor_sampling_seed') == seed and row.get('actor_sampling_draw_index') == index and
                row.get('actor_rng_before_sha256') == rng_sha(), 'actual Actor sampling seed/draw/RNG differs')
        sampled = int(torch.multinomial(probabilities[index:index+1], 1, replacement=True,
                                       generator=generator).item())
        require(sampled == row['action'] and row.get('actor_rng_after_sha256') == rng_sha(),
                'actual Actor sampled action/RNG differs')


def _phase(path):
    cfg=json.loads(Path(path).read_bytes())
    require(cfg.get('rpc_configuration_authoritative') is True and
            cfg.get('require_local_alp_audit') is False,'RPC-owned configuration/schema checks required')
    require(cfg.get('material_memory') == dict(contract=SCHEMA,memory_scope='whole_shift',
            tokenization_version=TOKENIZATION,reward_version=MATERIAL_REWARD_VERSION),
            'explicit material-memory contract required')
    require(cfg.get('training_reward') == TRAINING_REWARD,
            'explicit combined training reward contract required')
    _training_contract(cfg)
    require(cfg.get('wait_diagnostics_contract') == WAIT_DIAGNOSTICS_CONTRACT,
            'explicit terminal waiting diagnostics contract required')
    require(isinstance(cfg.get('recording_id'),str) and cfg['recording_id'] and
            isinstance(cfg.get('jobs'),list) and bool(cfg['jobs']),'explicit recording and jobs required')
    runs,seeds,groups=set(),{},{}
    from python.reward_config import REQUIRED
    for job in cfg['jobs']:
        require(all(type(job.get(k)) is int and 0<job[k]<2**63
                    for k in ('run_index','experiment_seed','actor_sampling_seed')),'explicit positive run/environment/action seeds required')
        require(job['run_index'] not in runs,'duplicate configured run identity')
        runs.add(job['run_index'])
        partition=job.get('partition','synthetic')
        require(partition in ('train','validation','synthetic'),'unknown configured partition')
        for kind in ('experiment_seed','actor_sampling_seed'):
            key=(kind,job[kind])
            require(key not in seeds or seeds[key]==partition,'train/validation seed leakage')
            seeds[key]=partition
        if job.get('scene_group'):
            group=job['scene_group']
            require(group not in groups or groups[group]==partition,'train/validation scene group leakage')
            groups[group]=partition
        require(isinstance(job.get('expected_rpc_config'),dict) and
                {'config.'+key for key in REQUIRED}<=set(job['expected_rpc_config']),
                'complete explicit expected RPC configuration required')
        require(all(job['expected_rpc_config'].get(key) == value for key,value in
                    (('config.w_production',4.),('config.waste_shortfall_weight',4.),
                     ('config.wait_kappa',4e-5))),
                'raw RPC 4/4 production and waiting coefficient must remain unchanged')
        require_wait_rpc_configuration(job['expected_rpc_config'])
        if 'runtime_scene_sha256' in job:
            from python.runtime_scenario import PREFIX,fingerprint,validate_runtime_fields
            scene={k:v for k,v in job['expected_rpc_config'].items() if k.startswith(PREFIX)}
            validate_runtime_fields(scene,scene)
            require(job['runtime_scene_sha256']==fingerprint(scene),'runtime scene identity hash differs from frozen fields')
    from .features import feature_contract
    require(cfg.get('feature_contract') == feature_contract(), 'decision feature contract differs')
    return cfg


def _validated_wait_deltas(infos):
    """Reconstruct the finite nonnegative per-transition fleet-wait ledger.

    Raw production-v2 accounting remains independently validated by the inherited
    loader. These explicit checks bind the new training component to each RPC
    settlement rather than distributing the terminal total across decisions.
    """
    require(len(infos) >= 2, 'complete RPC wait ledger required')
    start=infos[0]['reward_components']
    require(start.get('wait_vehicle_seconds') == 0. and
            start.get('delta_wait_vehicle_seconds') == 0., 'unaccounted initial waiting')
    previous=0.;deltas=[]
    for info in infos[1:]:
        components=info['reward_components']
        require(all(key in components and type(components[key]) in (int,float) and
                    np.isfinite(components[key]) and components[key] >= 0. for key in
                    ('wait_vehicle_seconds','delta_wait_vehicle_seconds')),
                'finite nonnegative explicit RPC waiting components required')
        cumulative=float(components['wait_vehicle_seconds'])
        delta=float(components['delta_wait_vehicle_seconds'])
        require(cumulative >= previous and np.isclose(cumulative-previous,delta,rtol=1e-10,atol=1e-7),
                'RPC waiting increment/cumulative conservation differs')
        deltas.append(delta);previous=cumulative
    require(np.isclose(np.sum(deltas,dtype=np.float64),previous,rtol=1e-10,atol=1e-7),
            'RPC final fleet wait total differs from increments')
    return np.asarray(deltas,dtype=np.float64)


def _check_query_physics(events,states):
    """Rebind each causal material prefix to the existing raw 1519D snapshot."""
    from collections import deque
    fifo=deque();q=m=0.;index=0
    def close(a,b,name):
        require(np.isclose(a,b,rtol=2e-6,atol=2e-7),'RPC observation/material prefix differs: '+name)
    for event in events:
        if event.kind=='UNLOAD':fifo.append([event.batch_id,event.tons,event.grade])
        elif event.kind=='PROCESS':
            fifo[0][1]-=event.tons
            if fifo[0][1]<=1e-8:fifo.popleft()
            q+=event.tons;m+=event.metal
        elif event.kind=='WINDOW_CLOSE':q=m=0.
        elif event.kind in ('DECISION','END'):
            state=states[index];index+=1
            close(float(state[4]),q/5000,'window tons')
            close(float(state[5]),m/5000,'window metal')
            if event.kind=='DECISION':close(float(state[2]),event.window/4,'window index')
            for slot in range(16):
                start=1391+slot*3
                active=slot<len(fifo)
                close(float(state[start]),float(active),'FIFO validity')
                close(float(state[start+1]),fifo[slot][1]/500 if active else 0.,'FIFO tons')
                close(float(state[start+2]),fifo[slot][2] if active else 0.,'FIFO grade')
    require(index==len(states),'material query/state row count differs')


def admit_episode(raw_directory, journal_path, *, config_path, actor_path, output_path,
                  split, allow_synthetic=False):
    """Create one immutable derived HDF5 episode from existing sealed artifacts.

    This is data preparation only. Synthetic data is explicitly opted in and
    remains synthetic. Real data also reuses the inherited RPC/schema/scene
    admission path. Native rules cannot become fixed-Actor PPO trajectories.
    """
    import torch
    from python.constrained_batch import load_dataset
    from python.reward_config import REQUIRED,configuration_from_components
    from python.collect_offline import validate_runtime
    cfg = _phase(config_path)
    cfg_ref,actor_ref,journal_ref = reference(config_path),reference(actor_path),reference(journal_path)
    raw_ref=reference(Path(raw_directory)/'manifest.json')
    journal=read_journal(journal_path)
    header=journal.header
    require(header['recording_id']==cfg['recording_id'] and header['config_sha256']==cfg_ref['sha256'],
            'journal recording/configuration byte binding differs')
    synthetic=header['source_kind']=='synthetic'
    require(not synthetic or allow_synthetic,'synthetic journal requires explicit opt-in')
    rows,episodes,_,_,manifest=load_dataset(raw_directory,require_real=not synthetic)
    try:
        raw_payloads={str((Path(raw_directory)/name).resolve()):expected_sha
                      for name,expected_sha in manifest['files_sha256'].items()}
        require(all(reference(path)['sha256']==expected_sha for path,expected_sha in raw_payloads.items()),
                'sealed RPC payload bytes differ')
        require(len(episodes)==1,'one sealed source episode per material journal required')
        episode=episodes[0]
        require(manifest['source_kind']==('synthetic_unit_test' if synthetic else 'real_anylogic_rpc'),
                'RPC/journal source scope differs; native rules cannot enter fixed Actor data')
        actor,policy=frozen_actor(actor_path,allow_small_synthetic=synthetic)
        _check_actor_phase(actor, cfg)
        from .features import feature_contract
        require(cfg['feature_contract'] == feature_contract(actor.feature_config), 'Actor feature config differs from phase')
        require(episode['episode_id']==header['episode_id'] and episode['experiment_run_index']==header['run_index']
                and episode['experiment_seed']==header['seed'],'RPC/journal episode/run/seed differs')
        require(episode['policy_version']==policy and all(row['policy_version']==policy and
                not row.get('critic_only',False) and row.get('behavior_kind')!='anylogic_native' for row in rows),
                'fixed Actor/native provenance differs')
        selected=[j for j in cfg['jobs'] if j['run_index']==header['run_index']]
        require(len(selected)==1,'journal job must match exactly one configuration row')
        job=selected[0]
        require(job['experiment_seed']==header['seed'] and type(job.get('actor_sampling_seed')) is int
                and job['actor_sampling_seed']>0,'explicit matching environment/action seeds required')
        expected=job.get('expected_rpc_config')
        require(isinstance(expected,dict) and {'config.'+key for key in REQUIRED}<=set(expected),
                'complete explicit expected RPC configuration required')
        infos=[episode['start_info']]+[row['settlement_info'] for row in rows]
        wait_deltas=_validated_wait_deltas(infos)
        wait_breakdown=validate_rpc_wait_breakdown(infos)
        for info in infos:
            actual=configuration_from_components(info['reward_components'],terminal=info is infos[-1])
            require(actual.get('target_grade')==header['target_grade'] and
                    actual.get('grade_lower')==header['grade_lower'] and
                    actual.get('grade_upper')==header['grade_upper'],
                    'RPC grade configuration differs from journal header')
            require(all(k in actual.to_dict() and actual.to_dict()[k]==v for k,v in expected.items()),
                    'RPC differs from expected runtime configuration')
            if not synthetic: validate_runtime(info,job)
        if not synthetic:
            require(job.get('partition')==split and split in ('train','validation')
                    and isinstance(job.get('runtime_scene_sha256'),str),
                    'real episode needs frozen partition and runtime scene fingerprint')
            from python.runtime_scenario import PREFIX,fingerprint
            require(job['runtime_scene_sha256']==fingerprint({k:v for k,v in expected.items() if k.startswith(PREFIX)}),
                    'runtime scene identity hash differs from frozen fields')
            require(job.get('scene_id')==episode['scene_id'] and job.get('scene_group')==episode['scene_group'],
                    'RPC scene identity/group differs from configured partition')
            recorded=(manifest.get('provenance') or {}).get('configuration')
            require(recorded==cfg_ref,'real RPC source does not bind these exact configuration bytes')
        memory=build_memory(journal.events)
        require(len(memory.queries)==len(rows),'material query count differs from RPC transitions')
        markers=[e for e in journal.events if e.kind in ('DECISION','END')]
        for marker,info in zip(markers,infos):
            require(marker.request_id==info['request_id'] and marker.decision_sequence==info['decision_sequence']
                    and marker.time==info['sim_time_seconds'],'material marker/RPC callback identity or time differs')
        for settlement,window in zip(memory.settlements,episode['windows']):
            for observed,target,name in ((settlement.tons,window['ore_tons'],'tons'),
                    (settlement.metal,window['metal_tons'],'metal'),
                    (settlement.grade_reward_raw,window['grade_reward_raw'],'raw grade reward')):
                require(np.isclose(observed,target,rtol=1e-10,atol=1e-8),'material/RPC window '+name+' differs')
        states=np.stack([rows[0]['observation']]+[row['next_observation'] for row in rows]).astype(np.float32)
        _check_query_physics(journal.events,states)
        actions=np.asarray([row['action'] for row in rows],dtype=np.int64)
        log_probs=np.asarray([row['old_log_prob'] for row in rows],dtype=np.float64)
        with torch.no_grad():
            from .policy import replay_policy
            replay=replay_policy(actor, states[:-1], actions=actions)
            current=replay['log_probs'].numpy()
            if not synthetic: _verify_sampling(rows, replay['probabilities'], job)
        require(np.allclose(current,log_probs,rtol=1e-6,atol=1e-6),'old probabilities differ from the fixed Actor')
        terminal=rows[-1]['settlement_info']['reward_components']
        evaluation=dict(ore_tons=float(terminal['ore_tons_shift']),waste_tons=float(terminal['waste_tons_shift']),
            total_wait_seconds=float(terminal['wait_vehicle_seconds']),production_guarantee='pending',
            both_targets_met=bool(terminal['ore_tons_shift']>=16000 and terminal['waste_tons_shift']>=4000))
        evaluation.update(wait_breakdown)
        targets=build_composite_targets(memory,wait_deltas=wait_deltas,
            ore_tons=evaluation['ore_tons'],waste_tons=evaluation['waste_tons'],
            total_wait_seconds=evaluation['total_wait_seconds'],
            wait_weight=cfg['composite_contract']['wait_weight'])
        identity=dict(recording_id=header['recording_id'],episode_id=header['episode_id'],run_index=header['run_index'],
            environment_seed=header['seed'],actor_sampling_seed=job['actor_sampling_seed'],
            policy_version=policy,policy_sha256=actor_ref['sha256'],config_sha256=cfg_ref['sha256'],
            observation_schema='dispatch.physical.v3',reward_version=REWARD_VERSION,
            tokenization_version=TOKENIZATION,source_kind='synthetic' if synthetic else 'fixed_actor',
            split='synthetic' if synthetic else split,raw_manifest_sha256=raw_ref['sha256'],
            scene_id=episode['scene_id'],scene_group=episode['scene_group'],
            journal_file_sha256=journal_ref['sha256'],journal_logical_sha256=journal.logical_sha256,
            raw_process_calls=journal.raw_process_calls,
            decision_feature_version=cfg['feature_contract']['version'],
            decision_feature_config_sha256=cfg['feature_contract']['config_sha256'])
        def validate(derived):
            require(derived['identity']==identity,'derived identity differs from admitted artifacts')
            require(reference(config_path)==cfg_ref and reference(actor_path)==actor_ref and
                    reference(journal_path)==journal_ref and reference(Path(raw_directory)/'manifest.json')==raw_ref,
                    'source bytes changed during material admission')
            require(all(reference(path)['sha256']==expected_sha for path,expected_sha in raw_payloads.items()),
                    'sealed RPC payload changed during material admission')
            require(np.array_equal(derived['states'],states) and np.array_equal(derived['actions'],actions)
                    and np.array_equal(derived['old_log_probs'],log_probs),'derived Actor/state columns differ')
            require(derived['memory']==memory,'derived material tape differs')
            require(all(derived['evaluation'].get(k)==v for k,v in wait_breakdown.items()),
                    'derived waiting breakdown differs from terminal RPC')
            require(np.array_equal(derived['wait_deltas'],wait_deltas) and
                    derived['targets'] == targets and derived['composite_contract'] == cfg['composite_contract'],
                    'derived combined training cost ledger differs from RPC components')
        write_episode_hdf5(output_path,identity=identity,states=states,actions=actions,
            old_log_probs=log_probs,memory=memory,evaluation=evaluation,
            wait_deltas=wait_deltas,wait_weight=cfg['composite_contract']['wait_weight'],validator=validate)
        # The same admission callback rebinds source bytes and columns on read.
        checked=read_episode_hdf5(output_path,expected_identity=identity,validator=validate)
        return dict(status='PASS_DATA_PREPARATION_ONLY',output=reference(output_path),
            identity=identity,transitions=len(rows),events=len(memory.events),tokens=len(memory.tokens),
            actor_or_critic_fitted=False,collection_started=False,ppo_authorized=False,
            production_constraint_status='PENDING_BEFORE_ANY_ACTOR_OPTIMIZATION')
    finally:
        if hasattr(rows,'close'): rows.close()
