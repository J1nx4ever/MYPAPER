"""Immutable identities and artifacts for the four-strategy RPC validation."""
from hashlib import sha256
from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[3]
for directory in (ROOT, PLAN.parent / 'structural_ablation_001'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from baseline1_v1.java_plan import COLUMNS
from business_graph_summary_campaign_v1.contract import require
from business_graph_summary_campaign_v1.plan_io import encoded, load_reference, reference
from python.offline_split import episode_id
from python.runtime_scenario import PREFIX, fingerprint


STRATEGIES = ('unload_greedy', 'fifo_greedy', 'rolling', 'bg128')
SCHEMA = 'bg128-traditional-smoke-v1'
PHASE_SCHEMA = 'bg128-traditional-smoke-phase-v1'
CAMPAIGN = 'bg128-baseline-smoke-20261122-001'
FIRST_RUN = 10020001
PREPARED = PLAN / 'prepared_001'
SUITE = PREPARED / 'suite.json'
OUTPUT = ROOT / 'runs' / CAMPAIGN
REPORT = ROOT / 'reports/token_attention_001/business_graph_summary_009/baseline_comparison_001'
CONTROL = OUTPUT / 'control'
EVENTS = OUTPUT / 'anylogic_events'
METHOD = dict(planning_ms=800, depth=3, response_target_ms=1000)
EXECUTION = dict(host='127.0.0.1', port=50051, callback_timeout_seconds=3600)
PYTHON = '/home/jinx/miniconda3/envs/anylogic-env/bin/python'
CORE = ROOT.parents[1] / 'python/episode_budget_ppo.py'
CORE_SHA = '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'
SOURCE = ROOT / 'runs/bg128-continuous-mixed-20261119-001'
NOMINAL = SOURCE / 'evaluation/nominal/new'
MODEL = SOURCE / 'arms/s01_BG128/updates/0032'
NOMINAL_SHA = 'db572de9fe90d1081088f6000c9d281c60718831273746ecdd2054e8f74e1422'
CHECKPOINT_SHA = 'b8b81c0821535cdfdff3c9352c11d69bc91d14fbe97694ab5f45c08fb97c3c9a'
ACTOR_SHA = '99cf52513539491f1e2076b57efa78350a52dcf7bbd2225ad2b725ece95a77c8'
LAYOUT = dict(python_output=str(OUTPUT), control_directory=str(CONTROL), report_directory=str(REPORT),
              event_root=str(EVENTS))


def source_pins():
    strategy = ROOT / 'traditional_dispatch_v1'
    strategy_sources = sorted(p for p in strategy.glob('*.py') if not p.name.startswith('test_'))
    require({'runtime.py', 'decision.py'} <= {p.name for p in strategy_sources},
            'finish the baseline runtime and decision sources before preparing')
    return [reference(path) for path in [*sorted(PLAN.glob('*.py')), *strategy_sources,
                PLAN/'methods.md', PLAN/'single_shift_anylogic.md',
                ROOT/'python/observation_schema.json', ROOT/'python/dispatch_bridge.py',
                ROOT/'python/dispatch_env.py', ROOT/'python/constrained_adapter.py',
                ROOT/'python/runtime_scenario.py', ROOT/'physical_realization_v1/observation.py',
                ROOT/'single_critic_task_query_v1/targets.py']]


def source_contract():
    refs = dict(protected_core=reference(CORE), nominal_source=reference(NOMINAL/'phase_config.json'),
                checkpoint=reference(MODEL/'checkpoint.pt'), actor=reference(NOMINAL/'actor.pt'),
                actor_manifest=reference(NOMINAL/'actor.pt.manifest.json'), source_completion=reference(SOURCE/'suite_complete.json'),
                checkpoint_manifest=reference(MODEL/'checkpoint.pt.manifest.json'),
                checkpoint_seal=reference(MODEL/'checkpoint.pt.seal.json'))
    for key, expected in [('protected_core', CORE_SHA), ('nominal_source', NOMINAL_SHA),
                          ('checkpoint', CHECKPOINT_SHA), ('actor', ACTOR_SHA)]:
        require(refs[key]['sha256'] == expected, 'frozen source differs: ' + key)
    nominal = load_reference(refs['nominal_source'])
    template = nominal['jobs'][0]
    rpc = dict(template['expected_rpc_config'], **{'config.material_event_memory_enabled': 0.0,
                                                 'config.wait_breakdown_enabled': 1.0,
                                                 'config.wait_breakdown_version': 1.0})
    scene = fingerprint({key: val for key, val in rpc.items() if key.startswith(PREFIX)})
    require(scene == template['runtime_scene_sha256'], 'source physical fingerprint differs')
    completion = load_reference(refs['source_completion'])
    require(completion['status'] == 'COMPLETE' and completion['new_checkpoint'] == refs['checkpoint']
            and completion['training_updates'] == 32 and completion['evaluation_optimizer_updates'] == 0,
            'frozen mixed training or fixed evaluation completion differs')
    refs['source_suite'] = completion['campaign']
    learner = load_reference(refs['source_suite'])['learner_config']
    require(learner['seed'] == template['actor_seed'], 'source checkpoint initialization differs')
    return refs, learner, rpc, scene


def seed(stream):
    return int.from_bytes(sha256(f'{CAMPAIGN}/{stream}/0'.encode('ascii')).digest()[:6], 'big')


def job_for(ordinal, environment_seed, actor_sampling_seed, learner, rpc, scene):
    policy = STRATEGIES[ordinal]
    actor = policy == 'bg128'
    current_output, event_directory = OUTPUT / policy, EVENTS / policy
    recording = f'{CAMPAIGN}-{policy}'
    run = FIRST_RUN + ordinal
    return dict(ordinal=ordinal, actor_seed=learner['seed'] if actor else 0,
                group='BASELINE', role='final', source_recording_id=recording,
                recording_id=recording, run_index=run, experiment_seed=environment_seed,
                actor_sampling_seed=actor_sampling_seed if actor else 0,
                actor_sampling_seed_consumed=actor, event_directory=str(event_directory),
                event_bundle=str(event_directory / episode_id(run, environment_seed)),
                python_output=str(current_output), target_training_update=0,
                checkpoint_update=32 if actor else 0, policy_kind=policy, scene_id='nominal',
                expected_rpc_config=rpc, runtime_scene_sha256=scene)


def config_for(job, refs):
    body = {key: value for key, value in job.items() if key != 'phase_config'}
    cfg = dict(schema=PHASE_SCHEMA, campaign_id=CAMPAIGN, policy_kind=job['policy_kind'],
               job=body, method=METHOD, execution=EXECUTION, rpc_configuration_authoritative=True,
               require_local_alp_audit=False, training_shifts=0, optimizer_updates=0,
               capture_physical_events=False, capture_material_events=False,
               diagnostics=dict(rpc_jsonl=False, transition_jsonl=False, rollout_snapshots=False,
                                capture_observations=False), protected_core=refs['protected_core'])
    if job['policy_kind'] == 'bg128':
        cfg.update({key: refs[key] for key in ('actor', 'checkpoint', 'checkpoint_manifest', 'checkpoint_seal')})
    return cfg


def validate_jobs(jobs):
    require(len(jobs) == len(STRATEGIES) and tuple(row['policy_kind'] for row in jobs) == STRATEGIES,
            'four frozen strategies in order required')
    for key in ('run_index', 'recording_id', 'event_directory', 'event_bundle', 'python_output'):
        require(len({row[key] for row in jobs}) == len(STRATEGIES), 'duplicate physical ' + key)
    env, sample = jobs[0]['experiment_seed'], jobs[-1]['actor_sampling_seed']
    require(env == seed('environment') and sample == seed('actor-sampling')
            and type(env) is int and type(sample) is int and 0 < env < 2**48 and 0 < sample < 2**48 and env != sample,
            'two fresh, distinct positive Java-width seeds required')
    require(all(row['experiment_seed'] == env for row in jobs), 'physical pairing must use one environment seed')
    require(all(row['actor_sampling_seed'] == 0 and row['actor_seed'] == 0
                and row['actor_sampling_seed_consumed'] is False for row in jobs[:-1])
            and jobs[-1]['actor_sampling_seed_consumed'] is True,
            'only BG128 receives a private sampling stream')


def load_suite(path=SUITE, *, check_sources=True):
    require(Path(path).resolve() == SUITE, 'the single frozen smoke suite path is required')
    suite = load_reference(reference(path))
    require(Path(path).read_bytes() == encoded(suite), 'noncanonical smoke suite bytes')
    require(suite['schema'] == SCHEMA and suite['campaign_id'] == CAMPAIGN and suite['execution'] == EXECUTION
            and suite['method'] == METHOD and all(suite.get(key) == value for key, value in LAYOUT.items()),
            'smoke suite contract differs')
    require(suite['pairing'] == dict(scope='this_campaign_only', slots=1, share='experiment_seed',
                                    policies=list(STRATEGIES)) and suite['physical_shifts'] == len(STRATEGIES)
            and suite['training_shifts'] == suite['optimizer_updates'] == 0,
            'smoke suite pairing or scope differs')
    jobs = suite['jobs']
    validate_jobs(jobs)
    refs, learner, rpc, scene = source_contract()
    require(suite['learner_config'] == learner and all(suite.get(key) == val for key, val in refs.items()),
            'frozen source contract differs')
    for i, job in enumerate(jobs):
        actual = {key: value for key, value in job.items() if key != 'phase_config'}
        require(actual == job_for(i, jobs[0]['experiment_seed'], jobs[-1]['actor_sampling_seed'],
                                  learner, rpc, scene), 'frozen physical job differs')
        ref = job['phase_config']
        path = PREPARED/'configs'/f"{job['policy_kind']}.json"
        require(reference(path) == ref and path.read_bytes() == encoded(config_for(job, refs)),
                'frozen per-strategy configuration bytes differ')
    for key, name in [('master_plan', 'master_plan.csv'), ('scenario_table', 'scenario_table.csv'),
                      ('java_helper', 'Baseline1CampaignPlan.java')]:
        require(reference(PREPARED / name) == suite[key], 'frozen launch artifact differs: ' + key)
    if check_sources:
        require(suite['source_pins'] == source_pins(), 'smoke implementation source bytes differ')
    return suite
