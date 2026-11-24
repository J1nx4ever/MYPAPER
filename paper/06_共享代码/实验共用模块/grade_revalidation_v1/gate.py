"""Read-only gates for64 new shifts under the immutable wait0.1 u8 policy."""
import copy
import csv
import json
from pathlib import Path

from baseline1_v1 import gate as baseline
from baseline1_v1.prepare import _seed
from grade_multiobjective_ppo_v1.gate import source_pins as prior_pins
from grade_multiobjective_ppo_v1.budget_policy import check_collisions

ROOT, REPO, PROTECTED_SHA = baseline.ROOT, baseline.REPO, baseline.PROTECTED_SHA
encoded, reference, sha, require, load_reference = (baseline.encoded, baseline.reference,
    baseline.sha, baseline.require, baseline.load_reference)
PLAN = ROOT / 'change_plans/grade_wait01_revalidation_001'
CAMPAIGN_ID = 'grade-wait01-u8-revalidation-seed1-20261109-001'
EXPERIMENT_ID = CAMPAIGN_ID
SUITE_CONTRACT = 'grade-wait01-u8-fixed-evaluation-suite-v1'
PHASE_CONTRACT = 'grade-wait01-u8-fixed-evaluation-rpc-phase-v1'
RUN_COUNT = 64
PROFILE = 'formal_32ep_v1'
CANDIDATE = 'a1_actor_gru'
PHASE_ID = 'eval0064'
PURPOSE = 'fresh_seed_independent_validation_of_fixed_wait01_u8'
PARENT_PATH = ROOT / 'runs/grade-production-wait-finetune-seed1-20261109-001/arms/wait01/updates/0008/checkpoint.pt'
PARENT_SHA = 'd943d2016c7b459c4227902e594f8230bc8a5435d1c732e06aa2c8e16d4b83c8'
TEMPLATE_PATH = ROOT / 'change_plans/grade_production_wait_finetune_001/prepared_h2_released_001/configs/wait01_final_u0008.json'
REFERENCE_EVALUATION = ROOT / 'runs/grade-production-wait-finetune-seed1-20261109-001/arms/wait01/final_u0008/evaluation_complete.json'
TRAINING_REWARD = 'negative_combined_grade_production_wait_cost_v1'
ACCEPTANCE = dict(
    grade=dict(per_shift_four_windows=True, focus_windows=[2, 3], lower=0.26, upper=0.28),
    production=dict(ore_target_tons=16000.0, waste_target_tons=4000.0,
        waste_statistics=['minimum', 'p5', 'p10', 'rate_ge_4050t', 'rate_ge_4100t'],
        extra_waste_hard_threshold=None),
    critic=dict(metrics=['EV', 'MSE', 'Bias'], aggregation='state_equal_and_episode_equal',
        target='fixed_composite_MC_cost', bias_sign='prediction_minus_target', hard_thresholds=None))


def parent_references():
    require(sha(PARENT_PATH) == PARENT_SHA, 'fixed wait0.1 u8 checkpoint bytes changed')
    return {key: reference(path) for key, path in dict(checkpoint=PARENT_PATH,
        manifest=Path(str(PARENT_PATH) + '.manifest.json'),
        seal=Path(str(PARENT_PATH) + '.seal.json')).items()}


def parent_configuration():
    """Read the sealed original configuration; never construct or reset a model."""
    from grade_multiobjective_ppo_v1.learner import (
        CHECKPOINT_SCHEMA, _hash, _load_checkpoint, implementation_signature, validate_config)
    parent_references()
    saved, manifest = _load_checkpoint(PARENT_PATH)
    cfg = validate_config(saved['config'])
    require(saved['schema'] == manifest['schema'] == CHECKPOINT_SCHEMA and
        saved['update_index'] == manifest['update_index'] == 8 and
        saved['config_sha256'] == manifest['config_sha256'] == _hash(cfg) and
        saved['implementation'] == implementation_signature() and
        cfg['seed'] == 2026110903 and cfg['candidate_id'] == CANDIDATE and
        cfg['execution_profile'] == PROFILE and cfg['reward_contract']['wait_weight'] == 0.1,
        'fixed sealed wait0.1 u8 learner/model implementation differs')
    return copy.deepcopy(cfg)


def source_pins():
    pins = prior_pins()
    pins.update({str(path.resolve()): sha(path) for path in (ROOT / 'grade_revalidation_v1').glob('*.py')
        if not path.name.startswith('test_')})
    return dict(sorted(pins.items()))


def check_sources(pins):
    require(pins == source_pins(), 'evaluation source inventory changed; freeze a new package')
    require(sha(REPO / 'python/episode_budget_ppo.py') == PROTECTED_SHA, 'protected PPO core changed')


def check_phase(cfg):
    from grade_multiobjective_ppo_v1.targets import composite_contract
    from grade_multiobjective_ppo_v1.features import feature_contract
    from grade_only_ppo_v1.gate import MEMORY
    template = json.loads(TEMPLATE_PATH.read_bytes())
    template_job = template['jobs'][0]
    require(cfg.get('contract') == PHASE_CONTRACT and cfg.get('kind') == 'single_seed_trial' and
        cfg.get('campaign_id') == CAMPAIGN_ID and
        cfg.get('experiment_id') == EXPERIMENT_ID and cfg.get('phase_id') == PHASE_ID and
        cfg.get('recording_id') == CAMPAIGN_ID + '-' + PHASE_ID and
        cfg.get('purpose') == PURPOSE and cfg.get('execution_profile') == PROFILE and
        cfg.get('candidate_id') == CANDIDATE and cfg.get('stage_id') == 'wait01' and
        cfg.get('evaluation_only') is True and cfg.get('optimizer_steps') == 0,
        'explicit independent fixed-policy evaluation phase required')
    compatible = copy.deepcopy(cfg)
    compatible['contract'] = baseline.PHASE_CONTRACT
    baseline.check_phase(compatible)
    require(cfg.get('material_memory') == MEMORY and cfg.get('feature_contract') == feature_contract() and
        cfg.get('composite_contract') == composite_contract(0.1) and
        cfg.get('training_reward') == TRAINING_REWARD and
        cfg.get('history_contract') == dict(scope='whole_shift', window_reset=False, terminal_reset=True,
            source='causal_decision_summaries', critic='decision_gru'),
        'original causal input/history and8/8 wait0.1 reward contracts required')
    require(cfg['reward']['kappa'] == 4e-5 and cfg['reward']['ore_weight'] ==
        cfg['reward']['waste_weight'] == 4 and cfg.get('arm_update') == 8 and
        cfg.get('arm_phase_id') == PHASE_ID,
        'raw RPC4/4 and immutable update8 identity must remain unchanged')
    require(all(cfg.get(key) == template[key] for key in
        ('reward', 'environment_acceptance', 'execution', 'storage', 'capture_physical_events',
         'diagnostic_recording', 'material_memory', 'feature_contract', 'composite_contract', 'history_contract')),
        'original physical, RPC, causal input and reward contracts must remain unchanged')
    output = ROOT / 'runs' / CAMPAIGN_ID
    legacy = output / 'control/legacy_unused' / PHASE_ID
    materials = ROOT / 'runs' / (CAMPAIGN_ID + '_materials') / PHASE_ID
    require(Path(cfg['python_output']) == output / 'evaluation' and Path(cfg['event_directory']) == legacy and
        len(cfg['jobs']) == RUN_COUNT, 'dedicated64-shift evaluation output required')
    for ordinal, job in enumerate(cfg['jobs']):
        run = 9050001 + ordinal
        seed = _seed(CAMPAIGN_ID, 'environment', ordinal)
        episode = f'dispatch-production-wait-shortfall-v2-run-{run}-seed-{seed}'
        group = CAMPAIGN_ID + f'-r{ordinal:04d}'
        require(job['ordinal'] == ordinal and job['run_index'] == run and job['experiment_seed'] == seed and
            job['actor_sampling_seed'] == _seed(CAMPAIGN_ID, 'actor', ordinal) and
            job['actor_seed'] == 2026110903 and job.get('candidate_id') == CANDIDATE and
            job.get('stage_id') == 'wait01' and job.get('arm_phase_id') == PHASE_ID and
            job.get('arm_update') == 8 and job['role'] == 'final' and job['group'] == 'BASELINE' and
            job['partition'] == 'validation' and job['target_training_update'] == 0 and
            job['checkpoint_update'] == 8 and job['source_recording_id'] == cfg['recording_id'] and
            all(job.get(key) == template_job[key] for key in
                ('expected_rpc_config', 'scene_id', 'runtime_scene_sha256')) and
            job['scene_group'] == job['realization_group_id'] == group and
            Path(job['event_directory']) == legacy and Path(job['event_bundle']) == legacy / episode and
            Path(job['material_event_directory']) == materials and
            Path(job['material_event_bundle']) == materials / episode and
            Path(job['material_journal']) == materials / episode / 'material_events.bin',
            'fresh validation identity, causal material path or fixed update differs')
    require(len({j['actor_sampling_seed'] for j in cfg['jobs']}) == RUN_COUNT and
        {j['experiment_seed'] for j in cfg['jobs']}.isdisjoint(j['actor_sampling_seed'] for j in cfg['jobs']),
        'independent environment/action seeds required')
    return cfg


def check_approval(approval, suite, suite_sha256):
    require(approval.get('decision') == 'approved' and approval.get('approved_by') == 'user' and
        approval.get('execution_authorized') is True and approval.get('ack_checkpoint_read_only') is True and
        bool(approval.get('source_user_instruction')) and approval.get('campaign_id') == CAMPAIGN_ID and
        approval.get('experiment_id') == EXPERIMENT_ID and approval.get('execution_profile') == PROFILE and
        approval.get('campaign_sha256') == suite_sha256 and approval.get('max_new_shifts') == RUN_COUNT and
        approval.get('checkpoint') == suite['parent_artifacts']['checkpoint'] and
        approval.get('legacy_reserved_shifts') == approval.get('active_h2_reserved_shifts') == 0 and
        approval.get('budget_policy') == suite['budget_policy'] and
        approval.get('accounting') == 'additional_fixed_checkpoint_evaluation' and
        approval.get('maximum_total_after_registration') == suite['maximum_total_if_approved'],
        'exact64-shift independent evaluation and read-only checkpoint authorization required')
    return approval


def preflight(path, *, approval_path=None):
    """Check immutable artifacts/accounting without opening a listener or output."""
    from .budget import budget_snapshot, budget_state_reference
    path = Path(path).resolve()
    suite = json.loads(path.read_bytes())
    output = ROOT / 'runs' / CAMPAIGN_ID
    require(suite.get('contract') == SUITE_CONTRACT and suite.get('campaign_id') == CAMPAIGN_ID and
        suite.get('experiment_id') == EXPERIMENT_ID and suite.get('kind') == 'independent_evaluation' and
        suite.get('purpose') == PURPOSE and suite.get('execution_profile') == PROFILE and
        suite.get('new_shift_cap') == suite.get('evaluation_shifts') == RUN_COUNT and
        suite.get('training_shifts') == suite.get('training_updates') == suite.get('optimizer_steps') == 0 and
        suite.get('anylogic_start_count') == 1 and suite.get('actor_seed') == 2026110903 and
        suite.get('legacy_reserved_shifts') == suite.get('active_h2_reserved_shifts') == 0 and
        suite.get('parent_artifacts') == parent_references() and
        suite.get('learner_config') == parent_configuration() and suite.get('acceptance') == ACCEPTANCE,
        'exact64-shift fixed wait0.1 u8 evaluation suite required')
    require(Path(suite['python_output']) == output and Path(suite['control_directory']) == output / 'control' and
        Path(suite['evaluation_budget_path']) == ROOT / 'runs/independent-evaluation-budget-v1.json' and
        suite.get('logging') == 'compact_typed_hdf5_no_tick_jsonl',
        'dedicated evaluation output/control/budget and compact logging required')
    check_sources(suite['source_pins'])
    cfg = check_phase(load_reference(suite['phase_config']))
    from grade_multiobjective_ppo_v1.admission import _phase
    require(_phase(suite['phase_config']['path']) == cfg and cfg['jobs'] == suite['jobs'],
        'canonical independent phase jobs differ')
    require(reference(TEMPLATE_PATH) == suite['rpc_template_reference'], 'original RPC template changed')
    require(reference(REFERENCE_EVALUATION) == suite['reference_evaluation'] and
        suite.get('bootstrap_seed') == 2026110904 and suite.get('bootstrap_repetitions') == 10000,
        'predeclared descriptive uncertainty/reference report changed')
    from baseline1_v1.java_plan import COLUMNS
    for key in ('master_plan', 'java_helper'):
        require(reference(suite[key]['path']) == suite[key], key + ' bytes changed')
    with Path(suite['master_plan']['path']).open(newline='') as stream:
        require(list(csv.DictReader(stream)) == [{k: str(j[k]) for k in COLUMNS} for j in suite['jobs']],
            'canonical64-row master differs')
    from python.protocol_artifacts import verify_protocol_artifacts
    from python.collect_offline import verify_environment
    verify_protocol_artifacts()
    verify_environment(cfg['environment_acceptance'], cfg['reward'], storage_migration=True)
    engineering = load_reference(suite['engineering_acceptance'])
    require(engineering.get('status') == 'PASS_SYNTHETIC_ONLY' and
        engineering.get('source_pins') == suite['source_pins'] and
        all(engineering.get('gates', {}).get(key) == 'PASS_SYNTHETIC_ONLY' for key in ('G1', 'G2', 'G3', 'G4')),
        'matching64-shift synthetic engineering gates required')
    state, budget = budget_state_reference(), budget_snapshot()
    require(suite['budget_before'] == state['budget_before'] and suite['budget_policy'] == state['budget_policy'],
        'active budget history or H2 cancellation policy changed')
    registered = budget['campaigns'].get(CAMPAIGN_ID)
    if registered is None:
        require(suite.get('evaluation_budget_before') == state['evaluation_budget'],
            'independent evaluation budget changed since freeze')
        check_collisions(suite, budget)
        require(not output.exists() and not Path(str(output) + '_materials').exists(),
            'evaluation output identity already used')
    else:
        require(registered['campaign_sha256'] == sha(path), 'registered fixed evaluation suite changed')
    before = budget['maximum_total_shifts'] - (RUN_COUNT if registered else 0)
    require(suite['prior_baseline_reserved_shifts'] == before and
        suite['maximum_total_if_approved'] == before + RUN_COUNT,
        'additional64 cap must exclude cancelled H2 future reservation')
    approval = None if approval_path is None else check_approval(
        json.loads(Path(approval_path).read_bytes()), suite, sha(path))
    return suite, cfg, approval
