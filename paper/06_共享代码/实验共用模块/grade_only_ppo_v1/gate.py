"""Read-only preparation and explicit admission for a grade-only diagnostic PPO."""
import copy
import csv
import json
from pathlib import Path

from baseline1_v1 import gate as inherited
from baseline1_v1.budget import GLOBAL_PATH, _digest
from grade_event_memory_v1.events import SCHEMA
from grade_event_memory_v1.journal import TOKENIZATION
from grade_event_memory_v1.targets import REWARD_VERSION

ROOT, REPO = inherited.ROOT, inherited.REPO
PROTECTED_SHA = inherited.PROTECTED_SHA
encoded, reference, require, sha = inherited.encoded, inherited.reference, inherited.require, inherited.sha
load_reference = inherited.load_reference
CAMPAIGN_CONTRACT = 'grade-only-memory-campaign-v1'
PHASE_CONTRACT = 'grade-only-memory-rpc-phase-v1'
PLAN = ROOT / 'change_plans/grade_only_memory_ppo_001'
DIAGNOSTIC_SCOPE = dict(scope='grade_only_diagnostic', production='evaluation_only',
                        waiting='evaluation_only', production_guarantee=False)
MEMORY = dict(contract=SCHEMA, memory_scope='whole_shift',
              tokenization_version=TOKENIZATION, reward_version=REWARD_VERSION)
TRAINING_REWARD = 'negative_closed_window_grade_cost_only_no_wait_no_production'


def source_pins():
    pins = inherited.source_pins()
    paths = list((ROOT / 'grade_only_ppo_v1').glob('*.py'))
    paths += list((ROOT / 'grade_event_memory_v1').glob('*.py'))
    paths += [ROOT/'change_plans/grade_only_event_memory_engineering_001/java/MaterialEventRecorder.java']
    pins.update({str(p.resolve()): sha(p) for p in paths if not p.name.startswith('test_')})
    return dict(sorted(pins.items()))


def check_sources(pins):
    require(pins == source_pins(), 'grade-only implementation/source inventory changed; prepare new review')
    require(sha(REPO/'python/episode_budget_ppo.py') == PROTECTED_SHA, 'protected PPO core changed')


def budget_snapshot(path=None):
    """Validate a shared ledger without mkdir, lock files or mutation."""
    path = Path(path) if path is not None else GLOBAL_PATH
    data = json.loads(path.read_bytes())
    checksum = data.pop('document_sha256', None)
    require(checksum == _digest(data), 'global budget checksum mismatch')
    require(data.get('format') == 'baseline1-global-budget-v1'
            and data['legacy_reserved_shifts'] == 11280
            and data['additional_reserved_shifts'] == sum(v['cap'] for v in data['campaigns'].values()) == len(data['jobs'])
            and data['maximum_total_shifts'] == 11280 + data['additional_reserved_shifts'],
            'global budget accounting differs')
    return data


def check_collisions(campaign, budget):
    jobs, cid = campaign['jobs'], campaign['campaign_id']
    if cid in budget['campaigns']:
        return
    prior = list(budget['jobs'].values())
    for recovery in budget.get('recovery_history', {}).values():
        snapshot = load_reference(recovery['old_ledger_snapshot'])
        checksum = snapshot.pop('document_sha256', None)
        require(checksum == _digest(snapshot), 'retired budget checksum differs')
        prior.extend(snapshot['jobs'].values())
    for key in ('run_index', 'experiment_seed', 'actor_sampling_seed', 'recording_id'):
        old = {row['identity'].get(key) for row in prior} - {None}
        require(all(j.get(key) not in old for j in jobs), 'new identity collides with ledger history: '+key)


def check_phase(cfg):
    require(cfg.get('contract') == PHASE_CONTRACT, 'independent grade-only phase contract required')
    compatible = copy.deepcopy(cfg)
    compatible['contract'] = inherited.PHASE_CONTRACT
    inherited.check_phase(compatible)
    require(cfg.get('material_memory') == MEMORY, 'whole-shift causal material-memory contract required')
    require(cfg.get('diagnostic_scope') == DIAGNOSTIC_SCOPE, 'grade-only diagnostic scope must be explicit')
    require(cfg.get('training_reward') == TRAINING_REWARD, 'grade-only training reward differs')
    require(cfg['storage'] == dict(format='hdf5-v1', chunk_rows=64), 'compact HDF5 required')
    require(cfg['capture_physical_events'] is False and cfg['kind'] == 'single_seed_trial',
            'legacy full event/native collection must remain disabled')
    for job in cfg['jobs']:
        expected = job['expected_rpc_config']
        for key, value in dict(shift_seconds=28800., window_seconds=7200., target_ore_tons_shift=16000.,
                target_waste_tons_shift=4000., target_grade=.27, w_production=4., waste_shortfall_weight=4.,
                grade_lower=.26, grade_upper=.28, wait_kappa=.00004,
                material_event_memory_enabled=1., material_event_contract_version=1.).items():
            require(expected.get('config.'+key) == value, 'missing/different explicit RPC config: '+key)
        material_root = Path(job['material_event_directory'])
        bundle = Path(job['material_event_bundle'])
        require(material_root.is_absolute() and material_root.is_relative_to(ROOT/'runs')
                and bundle.parent == material_root and Path(job['material_journal']) == bundle/'material_events.bin',
                'material journal identity/path differs')
        episode = f"dispatch-production-wait-shortfall-v2-run-{job['run_index']}-seed-{job['experiment_seed']}"
        require(bundle.name == episode, 'material bundle must use the actual RPC episode ID')
        require(job['partition'] == ('train' if job['role'] == 'train' else 'validation'),
                'training and independent evaluation partitions differ')
    return cfg


def check_approval(approval, campaign, campaign_sha256):
    require(approval.get('decision') == 'approved' and approval.get('approved_by') == 'user'
            and approval.get('execution_authorized') is True and bool(approval.get('source_user_instruction'))
            and approval.get('ack_grade_only_diagnostic') is True
            and approval.get('campaign_id') == campaign['campaign_id']
            and approval.get('campaign_sha256') == campaign_sha256
            and approval.get('max_new_shifts') == campaign['new_shift_cap'] == 288
            and approval.get('legacy_reserved_shifts') == 11280
            and approval.get('accounting') == 'additional_to_h2_reserved'
            and approval.get('maximum_total_after_registration') == campaign['maximum_total_if_approved'],
            'exact separate budget/execution approval acknowledging grade-only diagnostic scope required')
    return approval


def preflight(campaign_path, *, approval_path=None):
    """No RPC, learner construction, ledger writes, output paths, or claims."""
    from .learner import validate_config
    from grade_event_memory_v1.admission import _phase
    path = Path(campaign_path).resolve()
    campaign = json.loads(path.read_bytes())
    require(campaign.get('contract') == CAMPAIGN_CONTRACT, 'grade-only memory campaign required')
    require(campaign['kind'] == 'single_seed_trial' and campaign['new_shift_cap'] == 288
            and campaign['legacy_reserved_shifts'] == 11280, 'fixed additional budget differs')
    require(campaign.get('diagnostic_scope') == DIAGNOSTIC_SCOPE, 'diagnostic-only scope differs')
    require(campaign.get('initialization') == 'fresh_random_actor_and_event_critic_no_pretraining',
            'fresh independent initialization required')
    check_sources(campaign['source_pins'])
    learner = validate_config(campaign['learner_config'])
    require(learner['source_kind'] == 'real_anylogic_rpc' and learner['seed'] == campaign['actor_seed'],
            'real campaign needs matching learner source and explicit initialization seed')
    output = ROOT/'runs'/campaign['campaign_id']
    require(Path(campaign['python_output']) == output and Path(campaign['control_directory']) == output/'control',
            'independent output/control identity differs')
    phases = [check_phase(load_reference(ref)) for ref in campaign['phase_configs']]
    for ref in campaign['phase_configs']: _phase(ref['path'])
    jobs = [j for p in phases for j in p['jobs']]
    require(jobs == campaign['jobs'] and len(jobs) == 288 and
            [j['ordinal'] for j in jobs] == list(range(288)), 'canonical jobs/ordinals differ')
    for key in ('run_index','experiment_seed','actor_sampling_seed'):
        require(len({j[key] for j in jobs}) == 288, 'duplicate global identity: '+key)
    partitions={}
    for job in jobs:
        group=job.get('scene_group')
        require(isinstance(group,str) and group, 'explicit scene realization group required')
        require(group not in partitions or partitions[group]==job['partition'],
                'training/independent evaluation scene-group leakage')
        partitions[group]=job['partition']
    expected = [('t0',0,16)] + [('train',i,32) for i in range(1,9)] + [('final',8,16)]
    require(len(phases) == len(expected), 'independent evaluation schedule differs')
    for phase, (role, update, count) in zip(phases, expected):
        require(phase['campaign_id'] == campaign['campaign_id'] and len(phase['jobs']) == count
                and phase['recording_id'] == campaign['campaign_id']+'-'+phase['phase_id']
                and Path(phase['python_output']) == output/phase['phase_id']
                and Path(phase['event_directory']) == output/'control'/'legacy_unused'/phase['phase_id'],
                'phase identity/count/path differs')
        for job in phase['jobs']:
            require(job['role'] == role and job['target_training_update'] == update
                    and job['checkpoint_update'] == (update-1 if role == 'train' else update)
                    and job['actor_seed'] == campaign['actor_seed']
                    and job['source_recording_id'] == phase['recording_id']
                    and job['phase_id'] == phase['phase_id'], 'job checkpoint/timeline differs')
            require(Path(job['material_event_directory']) == ROOT/'runs'/(campaign['campaign_id']+'_materials')/phase['phase_id'],
                    'material root differs from independent phase identity')
    require(len({j['recording_id'] for j in jobs}) == 10, 'independent per-phase recordings required')
    for key in ('master_plan','java_helper'):
        require(sha(campaign[key]['path']) == campaign[key]['sha256'], key+' bytes changed')
    from baseline1_v1.java_plan import COLUMNS
    with Path(campaign['master_plan']['path']).open(newline='') as stream: master = list(csv.DictReader(stream))
    require(master == [{key:str(j[key]) for key in COLUMNS} for j in jobs], 'master CSV differs from jobs')
    from python.protocol_artifacts import verify_protocol_artifacts
    from python.collect_offline import verify_environment
    verify_protocol_artifacts()
    for phase in phases: verify_environment(phase['environment_acceptance'], phase['reward'], storage_migration=True)
    acceptance = load_reference(campaign['engineering_acceptance'])
    require(acceptance.get('status') == 'PASS_SYNTHETIC_ONLY'
            and acceptance.get('source_pins') == campaign['source_pins']
            and all(acceptance.get('gates',{}).get(k) == 'PASS_SYNTHETIC_ONLY' for k in ('G1','G2','G3','G4')),
            'matching synthetic engineering G1-G4 acceptance required')
    budget = budget_snapshot()
    cid = campaign['campaign_id']
    if cid not in budget['campaigns']:
        require(reference(GLOBAL_PATH) == campaign['budget_before'], 'budget changed since preparation; reconcile first')
        check_collisions(campaign,budget)
        require(not output.exists() and all(not Path(j['material_event_directory']).exists() for j in jobs),
                'new run/material identity already used')
    else:
        require(budget['campaigns'][cid]['campaign_sha256'] == sha(path), 'registered campaign changed')
    require(campaign['prior_baseline_reserved_shifts'] ==
            budget['additional_reserved_shifts'] - (288 if cid in budget['campaigns'] else 0)
            and campaign['maximum_total_if_approved'] == 11280+campaign['prior_baseline_reserved_shifts']+288,
            'additional budget must preserve every prior allocation')
    approval = None if approval_path is None else check_approval(json.loads(Path(approval_path).read_bytes()),campaign,sha(path))
    return campaign, phases, approval
