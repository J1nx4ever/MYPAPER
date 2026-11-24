"""Identity and read-only gates for the new-seed fixed BG128 discovery batch."""
import csv
import importlib.util
import json
from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[4]
sys.path.insert(0, str(ROOT))

from baseline1_v1.java_plan import COLUMNS
from business_graph_summary_campaign_v1.plan_io import encoded, reference, sha, write
from business_graph_summary_evaluation_v1.gate import check_phase, check_sources, require, seed
from business_graph_summary_recovery_v1.admission import _phase
from physical_curriculum_v1.budget import _history_rows
from python.offline_split import java_random_states

CAMPAIGN = 'bg128-u13-discovery100-20261119-001'
PHASE = 's01_BG128_discover0100'
RECORDING = f'{CAMPAIGN}-{PHASE}'
COUNT = 100
FIRST_RUN = 9981001
OUTPUT = ROOT/'runs'/CAMPAIGN
MATERIALS = OUTPUT.with_name(CAMPAIGN+'_materials')/PHASE
CONTROL = OUTPUT/'control'
PREPARED = PLAN/'prepared_001'
SUITE = PREPARED/'suite.json'
APPROVAL = PREPARED/'approval.json'
REPORT = ROOT/'reports/token_attention_001/business_graph_summary_009/first_window_action_001/discovery_3e4_001'
OLD = PLAN.parents[1]/'checkpoint_evaluation_001/prepared_001/suite.json'
CORE = ROOT.parents[1]/'python/episode_budget_ppo.py'
CORE_SHA = '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'
CHECKPOINT_SHA = 'a9acc888413ce0861240c04c396a239d9c26e48290eddc9f2c14dd9c97e6afd5'
USER_INSTRUCTION = '继续保留3e-4/u13，不修改 Reward、GAT 或 C128。先执行已经准备好的100班固定策略发现批次，扩大失败样本。'


def read(path):
    return json.loads(Path(path).read_bytes())


def checked(ref):
    path = Path(ref['path'])
    require(reference(path) == ref, 'sealed bytes differ: ' + str(path))
    return path


def previous_budget():
    # The predecessor script also imports its own common/budget names.
    path = PLAN.parent/'followup_001/merged_budget.py'
    spec = importlib.util.spec_from_file_location('bg128_previous_budget', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.budget_snapshot()


def check_fresh_jobs(jobs, previous):
    require(CAMPAIGN not in previous['campaigns'], 'discovery campaign already registered')
    prior = _history_rows(previous)
    run_indices = {r.get('run_index') for r in prior}
    recordings = {r.get(k) for r in prior for k in ('recording_id', 'source_recording_id')}
    old_seeds = {r.get(k) for r in prior for k in ('experiment_seed','actor_seed','actor_sampling_seed')}
    old_seeds = {s for s in old_seeds if type(s) is int}
    # java.util.Random masks its initialized seed to 48 bits.
    mask = (1 << 48) - 1
    old_effective = {s & mask for s in old_seeds}
    fresh = [j[k] for j in jobs for k in ('experiment_seed','actor_sampling_seed')]
    require(len(fresh) == 2*COUNT and len(set(fresh)) == len(fresh) and
            all(type(s) is int and 0 < s <= mask for s in fresh) and
            not old_effective.intersection(fresh), 'history/new seed stream or effective 48-bit collision')
    require(RECORDING not in recordings and all(j['run_index'] not in run_indices for j in jobs),
            'old run or recording identity reused')
    require(all(j['actor_seed'] not in fresh for j in jobs), 'initialization and new sampling stream collide')
    previous_java = {state for j in prior if type(j.get('experiment_seed')) is int
                     for state in java_random_states(j['experiment_seed']).values()}
    new_java = [state for j in jobs for state in java_random_states(j['experiment_seed']).values()]
    require(len(set(new_java)) == len(new_java) == 5*COUNT and
            not previous_java.intersection(new_java), 'engine/site Java RNG initial state collision')


def check_package():
    suite = read(SUITE)
    old = read(checked(suite['original_suite']))
    require(checked(suite['original_suite']) == OLD, 'expected original 3e-4/u13 suite')
    require(suite['schema'] == 'bg128-u13-first-window-discovery-v1' and
            suite['campaign_id'] == CAMPAIGN and suite['checkpoint_update'] == 13 and
            suite['new_shift_cap'] == suite['evaluation_shifts'] == COUNT and
            suite['optimizer_steps'] == suite['training_updates'] == suite['training_shifts'] == 0,
            'discovery scope differs')
    require(suite['parent_artifacts'] == old['parent_artifacts'] and
            suite['parent_artifacts']['checkpoint']['sha256'] == CHECKPOINT_SHA and
            suite['learner_config'] == old['learner_config'] and
            suite['sampling'] == old['sampling'] == 'stochastic_private_rng' and
            suite['cpu_threads'] == old['cpu_threads'] == 2, 'original frozen learner/sampling differs')
    check_sources(suite['source_pins'])
    for ref in suite['parent_artifacts'].values():
        checked(ref)
    for key in ('selection','training_suite','training_completion','master_plan','java_helper','protocol'):
        checked(suite[key])
    for ref in suite['discovery_sources'].values():
        checked(ref)
    require(sha(CORE)==CORE_SHA and suite['python_output']==str(OUTPUT) and
            suite['control_directory']==str(CONTROL), 'core or output identity differs')
    cfg = _phase(checked(suite['phase_config']))
    check_phase(cfg)
    previous_cfg = _phase(checked(old['phase_config']))
    phase_changes = {'campaign_id','phase_id','arm_phase_id','recording_id','python_output','event_directory','jobs'}
    require({k:v for k,v in cfg.items() if k not in phase_changes} ==
            {k:v for k,v in previous_cfg.items() if k not in phase_changes}, 'original RPC/reward/network phase differs')
    jobs = cfg['jobs']
    require(jobs==suite['jobs'] and len(jobs)==COUNT and cfg['recording_id']==RECORDING and
            cfg['phase_id']==PHASE and cfg['python_output']==str(OUTPUT/'evaluation') and
            cfg['event_directory']==str(CONTROL/'legacy_unused'/PHASE), 'phase inventory or output differs')
    for ordinal, job in enumerate(jobs):
        run = FIRST_RUN+ordinal
        env = seed(CAMPAIGN,'environment',ordinal)
        bundle = f'dispatch-production-wait-shortfall-v2-run-{run}-seed-{env}'
        require((job['ordinal'], job['run_index'], job['experiment_seed'], job['actor_sampling_seed']) ==
                (ordinal,run,env,seed(CAMPAIGN,'sampling',ordinal)), 'run/seed derivation differs')
        require(job['recording_id']==job['source_recording_id']==RECORDING and
                job['material_event_directory']==str(MATERIALS) and job['material_event_bundle']==str(MATERIALS/bundle) and
                job['material_journal']==str(MATERIALS/bundle/'material_events.bin') and
                job['expected_rpc_config']==previous_cfg['jobs'][0]['expected_rpc_config'],
                'recording/material/config identity differs')
        require(job['actor_seed']==old['actor_seed'] and job['checkpoint_update']==13 and
                job['target_training_update']==0 and job['role']=='final' and
                job['partition']=='validation' and job['candidate_id']=='s01_BG128', 'job scope differs')
    with checked(suite['master_plan']).open(newline='') as handle:
        master = list(csv.DictReader(handle))
    require(master==[{k:str(j[k]) for k in COLUMNS} for j in jobs], 'Java master rows differ')
    approval = read(APPROVAL)
    require(approval==approval_for(suite), '100-shift user instruction and suite binding differ')
    return suite, cfg


def approval_for(suite):
    return dict(schema='bg128-discovery-authorization-v1',approved_by='user',decision='approved',
        source_user_instruction=USER_INSTRUCTION,campaign_id=CAMPAIGN,suite=reference(SUITE),
        checkpoint=suite['parent_artifacts']['checkpoint'],physical_shift_cap=COUNT,
        merged_maximum_after_registration=suite['maximum_total_if_approved'],optimizer_updates=0)
