"""Exact paired sampling modes, fixed policy, fresh seeds and budget gates."""
import copy
import csv
import json
from pathlib import Path

from baseline1_v1 import gate as baseline
from baseline1_v1.prepare import _seed
from grade_multiobjective_ppo_v1.budget_policy import check_collisions
from grade_revalidation_v1 import gate as previous

ROOT, REPO, PROTECTED_SHA = previous.ROOT, previous.REPO, previous.PROTECTED_SHA
encoded, reference, sha, require, load_reference = (previous.encoded, previous.reference,
    previous.sha, previous.require, previous.load_reference)
parent_references, parent_configuration = previous.parent_references, previous.parent_configuration
PARENT_PATH, PARENT_SHA = previous.PARENT_PATH, previous.PARENT_SHA
TEMPLATE_PATH = ROOT / 'change_plans/grade_wait01_revalidation_001/prepared/configs/eval0064.json'
PLAN = ROOT / 'change_plans/grade_wait01_sampling_ab_001'
CAMPAIGN_ID = 'grade-wait01-sampling-ab-seed1-20261109-001'
EXPERIMENT_ID = CAMPAIGN_ID
SUITE_CONTRACT = 'grade-wait01-fixed-checkpoint-sampling-ab-suite-v1'
PHASE_CONTRACT = 'grade-wait01-fixed-checkpoint-sampling-ab-rpc-phase-v1'
PROFILE, CANDIDATE = previous.PROFILE, previous.CANDIDATE
STAGES = ('stochastic', 'deterministic')
ACTION_SELECTION = dict(stochastic='categorical', deterministic='masked_argmax')
SEED_USAGE = dict(stochastic='consumed', deterministic='unused')
PAIR_COUNTS = (32, 64)
PURPOSE = 'paired_new_seed_categorical_vs_masked_argmax_fixed_wait01_u8'
ACCEPTANCE = dict(physical=copy.deepcopy(previous.ACCEPTANCE),
    comparison='paired_environment_seed_and_scene_group',
    changed_factor='evaluation_action_selection_only', checkpoint_selected_before_sampling=True,
    deterministic_tie_break='first_legal_maximum_index', training_authorized=False)


def source_pins():
    pins = previous.source_pins()
    pins.update({str(p.resolve()): sha(p) for p in (ROOT / 'grade_sampling_ab_v1').glob('*.py')
        if not p.name.startswith('test_')})
    return dict(sorted(pins.items()))


def check_sources(pins):
    require(pins == source_pins(), 'sampling comparison source inventory changed; freeze a new package')
    require(sha(REPO / 'python/episode_budget_ppo.py') == PROTECTED_SHA, 'protected PPO core changed')


def check_jobs(suite):
    count = suite.get('pair_count')
    require(type(count) is int and count in PAIR_COUNTS and
        suite.get('new_shift_cap') == 2 * count and len(suite.get('jobs', [])) == 2 * count,
        'exact32 or64 paired scenes and2N physical slots required')
    for stage_index, stage in enumerate(STAGES):
        phase = f'eval{count:04d}'
        phase_id = stage + '_' + phase
        for pair, job in enumerate(suite['jobs'][stage_index * count:(stage_index + 1) * count]):
            group = CAMPAIGN_ID + f'-paired-r{pair:04d}'
            require(job.get('ordinal') == stage_index * count + pair and job.get('pair_index') == pair and
                job.get('stage_id') == stage and job.get('phase_id') == phase_id and
                job.get('arm_phase_id') == phase and job.get('arm_update') == 8 and
                job.get('candidate_id') == CANDIDATE and job.get('actor_seed') == 2026110903 and
                job.get('run_index') == 9060001 + stage_index * 10000 + pair and
                job.get('experiment_seed') == _seed(CAMPAIGN_ID, 'environment', pair) and
                job.get('actor_sampling_seed') == _seed(CAMPAIGN_ID, 'actor', pair) and
                job.get('scene_group') == job.get('realization_group_id') == group and
                job.get('action_selection') == ACTION_SELECTION[stage] and
                job.get('actor_sampling_seed_usage') == SEED_USAGE[stage] and
                job.get('evaluation_only') is True and job.get('ppo_trainable') is False and
                job.get('role') == 'final' and job.get('partition') == 'validation' and
                job.get('group') == 'BASELINE' and job.get('target_training_update') == 0 and
                job.get('checkpoint_update') == 8 and
                job.get('recording_id') == job.get('source_recording_id') == CAMPAIGN_ID + '-' + phase_id,
                'paired scene identity, fixed checkpoint or explicit action-selection mode differs')
    require(len({job['run_index'] for job in suite['jobs']}) == 2 * count and
        len({job['experiment_seed'] for job in suite['jobs']}) == count and
        len({job['actor_sampling_seed'] for job in suite['jobs']}) == count and
        {job['experiment_seed'] for job in suite['jobs']}.isdisjoint(job['actor_sampling_seed'] for job in suite['jobs']),
        'only corresponding cross-mode pairs may share seeds')
    return suite['jobs']


def check_phase(cfg):
    from grade_multiobjective_ppo_v1.features import feature_contract
    from grade_multiobjective_ppo_v1.targets import composite_contract
    count = len(cfg.get('jobs', []))
    stage = cfg.get('stage_id')
    require(count in PAIR_COUNTS and stage in STAGES, 'explicit supported sampling comparison phase required')
    phase = f'eval{count:04d}'
    phase_id = stage + '_' + phase
    require(cfg.get('contract') == PHASE_CONTRACT and cfg.get('kind') == 'single_seed_trial' and
        cfg.get('campaign_id') == CAMPAIGN_ID and cfg.get('experiment_id') == EXPERIMENT_ID and
        cfg.get('phase_id') == phase_id and cfg.get('arm_phase_id') == phase and cfg.get('arm_update') == 8 and
        cfg.get('recording_id') == CAMPAIGN_ID + '-' + phase_id and cfg.get('purpose') == PURPOSE and
        cfg.get('candidate_id') == CANDIDATE and cfg.get('execution_profile') == PROFILE and
        cfg.get('action_selection') == ACTION_SELECTION[stage] and
        cfg.get('actor_sampling_seed_usage') == SEED_USAGE[stage] and cfg.get('evaluation_only') is True and
        cfg.get('ppo_trainable') is False and cfg.get('optimizer_steps') == 0,
        'fixed policy, zero-update or action-selection phase differs')
    compatible = copy.deepcopy(cfg)
    compatible['contract'] = baseline.PHASE_CONTRACT
    baseline.check_phase(compatible)
    template = json.loads(TEMPLATE_PATH.read_bytes())
    require(all(cfg.get(k) == template[k] for k in ('reward', 'environment_acceptance', 'execution', 'storage',
        'capture_physical_events', 'diagnostic_recording', 'material_memory', 'feature_contract',
        'composite_contract', 'history_contract', 'training_reward')) and
        cfg['feature_contract'] == feature_contract() and cfg['composite_contract'] == composite_contract(.1),
        'only action sampling changes; original RPC, reward, causal input and history must remain unchanged')
    output = ROOT / 'runs' / CAMPAIGN_ID
    legacy = output / 'control/legacy_unused' / phase_id
    material = ROOT / 'runs' / (CAMPAIGN_ID + '_materials') / stage / phase
    require(Path(cfg['python_output']) == output / 'arms' / stage / phase and
        Path(cfg['event_directory']) == legacy, 'dedicated mode evaluation paths required')
    for job in cfg['jobs']:
        episode = f"dispatch-production-wait-shortfall-v2-run-{job['run_index']}-seed-{job['experiment_seed']}"
        require(all(job.get(k) == template['jobs'][0][k] for k in
            ('expected_rpc_config', 'scene_id', 'runtime_scene_sha256')) and
            job['phase_id'] == phase_id and job['recording_id'] == cfg['recording_id'] and
            job['stage_id'] == stage and job['action_selection'] == ACTION_SELECTION[stage] and
            Path(job['event_directory']) == legacy and Path(job['event_bundle']) == legacy / episode and
            Path(job['material_event_directory']) == material and
            Path(job['material_event_bundle']) == material / episode and
            Path(job['material_journal']) == material / episode / 'material_events.bin',
            'physical environment, explicit mode or material path differs')
    return cfg


def check_approval(approval, suite, suite_sha256):
    require(approval.get('decision') == 'approved' and approval.get('approved_by') == 'user' and
        approval.get('execution_authorized') is True and approval.get('ack_checkpoint_read_only') is True and
        approval.get('ack_paired_sampling_modes') is True and bool(approval.get('source_user_instruction')) and
        approval.get('campaign_id') == CAMPAIGN_ID and approval.get('experiment_id') == EXPERIMENT_ID and
        approval.get('execution_profile') == PROFILE and approval.get('campaign_sha256') == suite_sha256 and
        approval.get('pair_count') == suite['pair_count'] and approval.get('max_new_shifts') == suite['new_shift_cap'] and
        approval.get('checkpoint') == suite['parent_artifacts']['checkpoint'] and
        approval.get('legacy_reserved_shifts') == approval.get('active_h2_reserved_shifts') == 0 and
        approval.get('budget_policy') == suite['budget_policy'] and
        approval.get('accounting') == 'additional_fixed_checkpoint_sampling_comparison' and
        approval.get('maximum_total_after_registration') == suite['maximum_total_if_approved'],
        'exact paired sampling-only comparison and read-only checkpoint authorization required')
    return approval


def preflight(path, *, approval_path=None):
    from .budget import budget_snapshot, budget_state_reference, GLOBAL_PATH
    path = Path(path).resolve()
    suite = json.loads(path.read_bytes())
    require(suite.get('contract') == SUITE_CONTRACT and suite.get('campaign_id') == CAMPAIGN_ID and
        suite.get('experiment_id') == EXPERIMENT_ID and suite.get('kind') == 'paired_sampling_evaluation' and
        suite.get('purpose') == PURPOSE and suite.get('execution_profile') == PROFILE and
        suite.get('arm_order') == list(STAGES) and suite.get('training_shifts') ==
        suite.get('training_updates') == suite.get('optimizer_steps') == 0 and
        suite.get('evaluation_shifts') == suite.get('new_shift_cap') and suite.get('anylogic_start_count') == 1 and
        suite.get('actor_seed') == 2026110903 and suite.get('parent_artifacts') == parent_references() and
        suite.get('learner_config') == parent_configuration() and suite.get('acceptance') == ACCEPTANCE and
        suite.get('legacy_reserved_shifts') == suite.get('active_h2_reserved_shifts') == 0,
        'exact fixed-checkpoint paired sampling comparison suite required')
    check_jobs(suite)
    check_sources(suite['source_pins'])
    output = ROOT / 'runs' / CAMPAIGN_ID
    require(Path(suite['python_output']) == output and Path(suite['control_directory']) == output / 'control' and
        Path(suite['sampling_budget_path']) == GLOBAL_PATH and suite.get('logging') == 'compact_typed_hdf5_no_tick_jsonl',
        'dedicated compact sampling-comparison outputs and budget required')
    require(len(suite['phase_configs']) == 2, 'two sampling mode configurations required')
    phases = [check_phase(load_reference(ref)) for ref in suite['phase_configs']]
    require([cfg['stage_id'] for cfg in phases] == list(STAGES) and
        [job for cfg in phases for job in cfg['jobs']] == suite['jobs'] and
        all(len(cfg['jobs']) == suite['pair_count'] for cfg in phases), 'canonical paired phase inventory differs')
    from .admission import _phase
    for ref, cfg in zip(suite['phase_configs'], phases):
        require(_phase(ref['path']) == cfg, 'sampling admission phase differs')
    require(reference(TEMPLATE_PATH) == suite['rpc_template_reference'], 'original independent RPC template changed')
    require(suite.get('bootstrap_seed') == 2026110905 and suite.get('bootstrap_repetitions') == 10000,
        'predeclared paired bootstrap settings changed')
    from baseline1_v1.java_plan import COLUMNS
    for key in ('master_plan', 'java_helper'):
        require(reference(suite[key]['path']) == suite[key], key + ' bytes changed')
    with Path(suite['master_plan']['path']).open(newline='') as stream:
        require(list(csv.DictReader(stream)) == [{key: str(job[key]) for key in COLUMNS} for job in suite['jobs']],
            'canonical paired master differs')
    from python.protocol_artifacts import verify_protocol_artifacts
    from python.collect_offline import verify_environment
    verify_protocol_artifacts()
    for cfg in phases:
        verify_environment(cfg['environment_acceptance'], cfg['reward'], storage_migration=True)
    engineering = load_reference(suite['engineering_acceptance'])
    require(engineering.get('status') == 'PASS_SYNTHETIC_ONLY' and engineering.get('source_pins') == suite['source_pins'] and
        all(engineering.get('gates', {}).get(k) == 'PASS_SYNTHETIC_ONLY' for k in ('G1', 'G2', 'G3', 'G4')),
        'matching paired sampling synthetic engineering gates required')
    state, budget = budget_state_reference(), budget_snapshot()
    require(all(suite.get(k) == state[k] for k in ('budget_before', 'revalidation_budget_before', 'budget_policy')),
        'preceding immutable budgets or H2 policy changed')
    registered = budget['campaigns'].get(CAMPAIGN_ID)
    if registered is None:
        require(suite.get('sampling_budget_before') == state['sampling_budget'], 'sampling budget changed since freeze')
        check_collisions(suite, budget)
        require(not output.exists() and not Path(str(output) + '_materials').exists(), 'sampling output identity already used')
    else:
        require(registered['campaign_sha256'] == sha(path), 'registered sampling comparison suite changed')
    cap = suite['new_shift_cap']
    before = budget['maximum_total_shifts'] - (cap if registered else 0)
    require(suite['prior_baseline_reserved_shifts'] == before and suite['maximum_total_if_approved'] == before + cap,
        'additional paired sampling cap must exclude cancelled H2 future reservation')
    approval = None if approval_path is None else check_approval(
        json.loads(Path(approval_path).read_bytes()), suite, sha(path))
    return suite, phases, approval
