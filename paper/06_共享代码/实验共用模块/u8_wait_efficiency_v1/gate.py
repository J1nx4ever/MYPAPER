"""Read-only gates for the fixed 812-shift u8 continuation comparison."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

from baseline1_v1 import gate as baseline
from grade_multiobjective_ppo_v1.gate import source_pins as historical_source_pins
from grade_only_ppo_v1.gate import MEMORY

ROOT, REPO, PROTECTED_SHA = baseline.ROOT, baseline.REPO, baseline.PROTECTED_SHA
encoded, reference, sha, require, load_reference = (baseline.encoded, baseline.reference,
    baseline.sha, baseline.require, baseline.load_reference)
PLAN = ROOT / 'change_plans/u8_wait_efficiency_001'
PREPARED_IDENTITIES = PLAN / 'helper_003/launch_identity.json'
CAMPAIGN_ID = EXPERIMENT_ID = 'u8-wait-efficiency-seed1-20261109-001'
SUITE_CONTRACT = CAMPAIGN_CONTRACT = 'u8-wait-efficiency-continuation-suite-v1'
PHASE_CONTRACT = 'u8-wait-efficiency-rpc-phase-v1'
PROFILE = 'formal_32ep_v1'
PURPOSE = 'paired_wait_efficiency_continuation_from_frozen_wait01_u8'
CANDIDATE = 'a1_actor_gru'
INITIALIZATION = 'immutable_wait01_u8_learned_parameters_and_full_adam_new_local_ledgers_rng'
TRAINING_REWARD = 'negative_combined_grade_production_wait_cost_v1'
STAGES = ('wait01', 'wait015', 'reference')
TRAIN_STAGES = ('wait01', 'wait015')
WAIT_WEIGHTS = dict(wait01=.1, wait015=.15, reference=.1)
PARENT_PATH = ROOT / 'runs/grade-production-wait-finetune-seed1-20261109-001/arms/wait01/updates/0008/checkpoint.pt'
PARENT_SHA = 'd943d2016c7b459c4227902e594f8230bc8a5435d1c732e06aa2c8e16d4b83c8'
TEMPLATE_PATH = ROOT / 'change_plans/grade_wait01_sampling_ab_001/prepared/configs/stochastic_eval0032.json'
COLUMNS = ('ordinal', 'actor_seed', 'group', 'role', 'source_recording_id', 'recording_id',
    'run_index', 'experiment_seed', 'actor_sampling_seed', 'event_directory', 'event_bundle',
    'python_output', 'target_training_update', 'checkpoint_update', 'stage_id', 'phase_id',
    'material_directory', 'wait_weight', 'pair_id', 'partition')
HISTORY_CONTRACT = dict(scope='whole_shift', window_reset=False, terminal_reset=True,
    source='causal_decision_summaries', critic='decision_gru')
DIAGNOSTIC_SCOPE = dict(scope='u8_wait_efficiency_paired_continuation',
    production='training_and_evaluation', waiting='original_fleet_wait_with_terminal_breakdown',
    production_guarantee=False, action_selection='categorical')
ACCEPTANCE = dict(evaluation_shifts_per_model=100, final_checkpoint_local_update=8,
    reference_checkpoint_parent_update=8, intermediate_evaluation_shifts=0,
    physical_task='four_windows_0.26_to_0.28_ore16000_waste4000',
    proposed_sample_successes=100, proposed_wait_reduction_fraction=.03,
    usage='report_prespecified_sample_criteria_not_population_guarantee_or_automatic_model_promotion')


def parent_references():
    require(sha(PARENT_PATH) == PARENT_SHA, 'immutable u8 parent bytes changed')
    return {key: reference(path) for key, path in dict(checkpoint=PARENT_PATH,
        manifest=Path(str(PARENT_PATH)+'.manifest.json'),
        seal=Path(str(PARENT_PATH)+'.seal.json')).items()}


def source_pins():
    pins = historical_source_pins()
    pins.update({str(path.resolve()): sha(path) for path in Path(__file__).parent.glob('*.py')
                 if not path.name.startswith('test_')})
    return dict(sorted(pins.items()))


def check_sources(pins):
    require(pins == source_pins(), 'u8 implementation changed; freeze a new package')
    require(sha(REPO/'python/episode_budget_ppo.py') == PROTECTED_SHA, 'protected PPO changed')


def schedule():
    phases = [(stage, 'train_u%04d' % update, 'train', update, 32)
              for stage in TRAIN_STAGES for update in range(1, 9)]
    phases += [(stage, 'final_0100', 'final', 0 if stage == 'reference' else 8, 100)
               for stage in ('reference', 'wait01', 'wait015')]
    return phases


def prepared_identities():
    value = json.loads(PREPARED_IDENTITIES.read_bytes())
    require(value.get('campaign_id') == CAMPAIGN_ID and len(value.get('jobs', [])) == 812,
            'reviewed identity preparation missing')
    return value


def canonical_master_row(job):
    return {key: str(job[key]) for key in COLUMNS}


def check_jobs(suite):
    jobs = suite.get('jobs')
    require(suite.get('campaign_id') == CAMPAIGN_ID and suite.get('new_shift_cap') == 812
            and isinstance(jobs, list) and len(jobs) == 812, 'exact 812-slot suite required')
    prepared = prepared_identities()
    position = 0
    for stage, name, role, local, count in schedule():
        qualified = stage+'_'+name
        for job, old in zip(jobs[position:position+count], prepared['jobs'][position:position+count]):
            expected = copy.deepcopy(old)
            expected['phase_id'] = qualified  # The shared collector requires job/config equality.
            require(all(str(job.get(key)) == str(expected[key]) for key in COLUMNS),
                    'run/seed/recording/master identity differs from reviewed preparation')
            require(job.get('candidate_id') == CANDIDATE and job.get('arm_phase_id') == name
                    and job.get('arm_update') == local and job.get('phase_id') == qualified
                    and job.get('action_selection') == 'categorical'
                    and job.get('evaluation_only') is (role == 'final')
                    and job.get('ppo_trainable') is (role == 'train')
                    and job.get('checkpoint_update') == (local-1 if role == 'train' else local),
                    'explicit stage/checkpoint/partition/action-mode scope differs')
            require(job.get('scene_group') == job.get('realization_group_id') ==
                    CAMPAIGN_ID+'-'+job['pair_id'], 'paired scene scope differs')
        position += count
    require(position == len(jobs) and len({job['run_index'] for job in jobs}) == 812,
            'duplicate or incomplete physical jobs')
    a, b = jobs[:256], jobs[256:512]
    r, ae, be = jobs[512:612], jobs[612:712], jobs[712:]
    paired = ('experiment_seed', 'actor_sampling_seed', 'pair_id', 'scene_group', 'partition')
    require(all(all(x[key] == y[key] for key in paired) for x, y in zip(a, b)),
            'training pairs differ')
    require(all(all(x[key] == y[key] == z[key] for key in paired)
                for x, y, z in zip(r, ae, be)), 'final evaluation pairs differ')
    train_seeds = {job[key] for job in a for key in ('experiment_seed', 'actor_sampling_seed')}
    final_seeds = {job[key] for job in r for key in ('experiment_seed', 'actor_sampling_seed')}
    require(len(train_seeds) == 512 and len(final_seeds) == 200
            and train_seeds.isdisjoint(final_seeds), 'training/final/cross-type seed leakage')
    return jobs


def check_phase(cfg):
    from .features import feature_contract
    from .targets import composite_contract
    from .waiting_diagnostics import WAIT_DIAGNOSTICS_CONTRACT, WAIT_RPC_FIELDS
    stage, name, local = cfg.get('stage_id'), cfg.get('arm_phase_id'), cfg.get('arm_update')
    require(stage in STAGES and cfg.get('contract') == PHASE_CONTRACT and
            cfg.get('campaign_id') == cfg.get('experiment_id') == CAMPAIGN_ID and
            cfg.get('candidate_id') == CANDIDATE and cfg.get('execution_profile') == PROFILE and
            cfg.get('purpose') == PURPOSE and cfg.get('phase_id') == stage+'_'+str(name),
            'new u8 RPC phase contract required')
    compatible = copy.deepcopy(cfg)
    compatible['contract'] = baseline.PHASE_CONTRACT
    baseline.check_phase(compatible)
    matched = [(s,n,r,u,c) for s,n,r,u,c in schedule() if (s,n,u) == (stage,name,local)]
    require(len(matched) == 1 and len(cfg['jobs']) == matched[0][4], 'phase schedule/count differs')
    role = matched[0][2]
    require(cfg.get('evaluation_only') is (role == 'final') and cfg.get('ppo_trainable') is (role == 'train')
            and cfg.get('action_selection') == 'categorical', 'phase train/evaluation separation differs')
    require(cfg.get('material_memory') == MEMORY and cfg.get('feature_contract') == feature_contract()
            and cfg.get('composite_contract') == composite_contract(WAIT_WEIGHTS[stage])
            and cfg.get('training_reward') == TRAINING_REWARD and cfg.get('history_contract') == HISTORY_CONTRACT
            and cfg.get('diagnostic_scope') == DIAGNOSTIC_SCOPE
            and cfg.get('wait_diagnostics_contract') == WAIT_DIAGNOSTICS_CONTRACT,
            'frozen input/reward/history/wait diagnostics differ')
    template = json.loads(TEMPLATE_PATH.read_bytes())
    require(all(cfg.get(key) == template[key] for key in ('reward', 'environment_acceptance', 'execution',
        'storage', 'capture_physical_events', 'diagnostic_recording')), 'physical/RPC/logging contract changed')
    expected = copy.deepcopy(template['jobs'][0]['expected_rpc_config'])
    expected.update(WAIT_RPC_FIELDS)
    for job in cfg['jobs']:
        episode = 'dispatch-production-wait-shortfall-v2-run-%d-seed-%d' % (job['run_index'], job['experiment_seed'])
        require(job['stage_id'] == stage and job['arm_update'] == local and job['arm_phase_id'] == name
            and job['recording_id'] == CAMPAIGN_ID+'-'+cfg['phase_id'] and job['role'] == role
            and job['expected_rpc_config'] == expected and
            all(job[key] == template['jobs'][0][key] for key in ('scene_id', 'runtime_scene_sha256'))
            and Path(job['material_event_directory']) == ROOT/'runs'/(CAMPAIGN_ID+'_materials')/stage/name
            and Path(job['material_event_bundle']) == Path(job['material_event_directory'])/episode
            and Path(job['material_journal']) == Path(job['material_event_bundle'])/'material_events.bin',
            'physical scene, original reward coefficients or material source binding differs')
    return cfg


def check_approval(approval, suite, suite_sha256):
    require(approval.get('decision') == 'approved' and approval.get('approved_by') == 'user'
        and approval.get('execution_authorized') is True and bool(approval.get('source_user_instruction'))
        and approval.get('campaign_id') == approval.get('experiment_id') == CAMPAIGN_ID
        and approval.get('campaign_sha256') == suite_sha256 and approval.get('max_new_shifts') == 812
        and approval.get('maximum_total_after_registration') == suite['maximum_total_if_approved']
        and approval.get('legacy_reserved_shifts') == approval.get('active_h2_reserved_shifts') == 0
        and approval.get('approved_stages') == list(STAGES)
        and approval.get('ack_wait_weights') == WAIT_WEIGHTS
        and approval.get('ack_full_adam_inheritance_no_reset') is True
        and approval.get('ack_immutable_u8_parent') is True
        and approval.get('ack_final_only_evaluation') is True
        and approval.get('ack_final_data_not_for_B_selection') is True,
        'explicit 812 shifts / 0.15 / inherited Adam / final-only evaluation approval required')
    return approval


def preflight(path, *, approval_path=None):
    from .budget import budget_snapshot, budget_state_reference, check_collisions
    from .learner import default_config, validate_config
    from .features import feature_contract
    from .targets import composite_contract
    from .waiting_diagnostics import WAIT_DIAGNOSTICS_CONTRACT
    from .admission import _phase
    path = Path(path).resolve()
    suite = json.loads(path.read_bytes())
    require(suite.get('contract') == SUITE_CONTRACT and suite.get('experiment_id') == CAMPAIGN_ID
        and suite.get('kind') == 'single_seed_trial' and suite.get('execution_profile') == PROFILE
        and suite.get('purpose') == PURPOSE and suite.get('initialization') == INITIALIZATION
        and suite.get('legacy_reserved_shifts') == suite.get('active_h2_reserved_shifts') == 0
        and suite.get('training_shifts') == 512 and suite.get('final_evaluation_shifts') == 300
        and suite.get('initial_evaluation_shifts') == 0 and suite.get('training_updates') == 16
        and suite.get('anylogic_start_count') == 1 and suite.get('arm_order') == list(STAGES)
        and suite.get('feature_contract') == feature_contract() and suite.get('acceptance_standard') == ACCEPTANCE
        and suite.get('stage_reward_contracts') == {stage:composite_contract(WAIT_WEIGHTS[stage]) for stage in STAGES}
        and suite.get('wait_diagnostics_contract') == WAIT_DIAGNOSTICS_CONTRACT
        and suite.get('diagnostic_scope') == DIAGNOSTIC_SCOPE
        and suite.get('final_evaluation_contract') == dict(strategy='freeze_all_before_any_final',pair_count=100,
            stages=['reference','wait01','wait015'],data_for_training=False,checkpoint_selection='fixed_local_update_8')
        and suite.get('parent_artifacts') == parent_references(), 'exact isolated u8 812 suite differs')
    check_jobs(suite)
    check_sources(suite['source_pins'])
    output = ROOT/'runs'/CAMPAIGN_ID
    require(Path(suite['python_output']) == output and Path(suite['control_directory']) == output/'control',
            'new dedicated output/control required')
    seed = prepared_identities()['initialization_rng_seed']
    require(suite.get('actor_seed') == seed and len(suite.get('arms', [])) == 3, 'common initialization differs')
    for arm, stage in zip(suite['arms'], STAGES):
        expected = default_config(seed=seed, candidate_id=CANDIDATE, wait_weight=WAIT_WEIGHTS[stage],
            source_kind='real_anylogic_rpc', execution_profile=PROFILE)
        require(arm['stage_id'] == stage and arm['actor_seed'] == seed and
            arm['candidate_id'] == CANDIDATE and Path(arm['python_output']) == output/'arms'/stage
            and validate_config(arm['learner_config']) == expected,
            'common u8 network, optimizer configuration or branch weight differs')
    phases = [check_phase(load_reference(ref)) for ref in suite['phase_configs']]
    require(len(phases) == 19 and [job for phase in phases for job in phase['jobs']] == suite['jobs'],
            'canonical phase/job order differs')
    for ref in suite['phase_configs']:
        _phase(ref['path'])
    for key in ('master_plan', 'java_helper', 'rpc_template_reference', 'prepared_identity_reference'):
        require(reference(suite[key]['path']) == suite[key], 'frozen '+key+' bytes changed')
    with Path(suite['master_plan']['path']).open(newline='') as handle:
        require(list(csv.DictReader(handle)) == [canonical_master_row(job) for job in suite['jobs']],
                'master must exactly match full runtime identities')
    helper = Path(suite['java_helper']['path']).read_text()
    require('private static final boolean PREPARATION_ONLY = false;' in helper and
        '"u8-wait-efficiency-ready-v1"' in helper and
        'protocol=baseline1-one-click-java-start-v1\\n' in helper,
        'execution helper ready/start protocols were not formally rendered')
    from python.protocol_artifacts import verify_protocol_artifacts
    from python.collect_offline import verify_environment
    verify_protocol_artifacts()
    for phase in phases:
        verify_environment(phase['environment_acceptance'], phase['reward'], storage_migration=True)
    acceptance = load_reference(suite['engineering_acceptance'])
    require(acceptance.get('status') == 'PASS_SYNTHETIC_ONLY' and
        acceptance.get('source_pins') == suite['source_pins'] and
        all(acceptance.get('gates', {}).get(key) == 'PASS_SYNTHETIC_ONLY' for key in ('G1','G2','G3','G4')),
        'matching continuation engineering validation required')
    budget = budget_snapshot()
    current = budget_state_reference()
    cid = suite['campaign_id']
    if cid not in budget['campaigns']:
        require(current == suite['budget_state_before'], 'budget changed since preparation')
        check_collisions(suite, budget)
        require(not output.exists() and not Path(str(output)+'_materials').exists(), 'output identity already used')
        prior = budget['maximum_total_shifts']
    else:
        require(budget['campaigns'][cid]['campaign_sha256'] == sha(path), 'registered suite changed')
        prior = budget['maximum_total_shifts']-812
    require(suite['maximum_total_if_approved'] == prior+812, 'exact newly approved capacity differs')
    approval = None if approval_path is None else check_approval(
        json.loads(Path(approval_path).read_bytes()), suite, sha(path))
    return suite, phases, approval
