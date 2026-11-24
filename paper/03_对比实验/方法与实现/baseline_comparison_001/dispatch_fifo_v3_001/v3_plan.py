"""Freeze a 100-slot FIFO DISPATCH evaluation paired to the formal V2 roster."""
import json
from pathlib import Path
import sys


PLAN = Path(__file__).resolve().parent
FORMAL = PLAN.parent/'paired_v2_100_001'
FLEET_TWO = PLAN.parents[1]/'fleet_outage_002'
for directory in (FORMAL, FLEET_TWO):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))

import formal_plan as previous
from formal_plan import (ROOT, CORE, EXECUTION, PYTHON, COLUMNS, PHASE_SCHEMA, smoke,
                         encoded, load_reference, reference, require)
from python.offline_split import JAVA_MASK, episode_id


SCHEMA = 'bg128-dispatch-fifo-v3-paired100-v1'
CAMPAIGN = 'bg128-dispatch-fifo-v3-paired100-20261122-002'
POLICY = 'dispatch_fifo_v3'
STRATEGIES = (POLICY,)
PREVIOUS_POLICY = 'dispatch_inspired'
SLOTS = COUNT = 100
FIRST_RUN = 10036001
METHOD = dict(planning_ms=800, replan_seconds=900, match_trucks=3, cohort_trial_limit=12, response_target_ms=1000)
PREPARED = PLAN/'prepared_002'
RETIRED_SUITE = dict(path=str(PLAN/'prepared_001/suite.json'),
                     sha256='04a36bf747033e07ab7bfe3be2673ccab8d6d27cff16d83c8128d6b13d9f32ff')
SUITE = PREPARED/'suite.json'
OUTPUT = ROOT/'runs'/CAMPAIGN
CONTROL = OUTPUT/'control'
EVENTS = OUTPUT/'anylogic_events'
REPORT = smoke.REPORT/'dispatch_fifo_v3_002'
REPORT_FILENAME = 'dispatch_fifo_v3_evaluation.json'
LAYOUT = dict(python_output=str(OUTPUT), control_directory=str(CONTROL), report_directory=str(REPORT),
              event_root=str(EVENTS))
PAIRING = dict(scope='formal_v2_slot_only', slots=SLOTS, share='experiment_seed', policies=list(STRATEGIES),
               previous_policy=PREVIOUS_POLICY, order='slot')


class PreviousV2Pending(RuntimeError):
    """The formal V2 controller has not published its final seal."""


def source_contract():
    formal = previous.load_suite()
    old = [job for job in formal['jobs'] if job['policy_kind'] == PREVIOUS_POLICY]
    require(len(old) == COUNT and all(job['pair_ordinal'] == slot for slot,job in enumerate(old))
            and len({job['experiment_seed'] for job in old}) == COUNT,
            'formal V2 DISPATCH pairing inventory differs')
    return formal, old


def algorithm_contract():
    from fifo_policy import FIFO_POLICY_CONTRACT
    from v3_flow import FLOW_PLAN_CONTRACT
    require(METHOD['cohort_trial_limit'] == FIFO_POLICY_CONTRACT['cohort_trial_limit'],
            'frozen cohort budget differs from the FIFO matching policy')
    return json.loads(encoded(dict(flow_plan_contract=FLOW_PLAN_CONTRACT, fifo_policy_contract=FIFO_POLICY_CONTRACT)))


def source_pins():
    from outage_plan import PLAN as fleet_plan
    prior, _ = source_contract()
    names = ('v3_plan.py', 'v3_ledger.py', 'prepare.py', 'v3_java.py', 'v3_flow.py', 'v3_matching.py',
             'fifo_policy.py', 'v3_runtime.py', 'launch.py')
    extra = (fleet_plan/'outage_ledger.py', fleet_plan/'outage_plan.py',
             PLAN.parent/'dispatch_inspired_001/dispatch_runtime.py', PLAN.parent/'paired_100_001/paired_report.py')
    shared = ('python/observation_contract.py', 'python/dispatch_service.py', 'python/episode_runner.py',
              'python/reward_config.py', 'python/reward_contract.py', 'python/scenario_contract.py',
              'python/generated/dispatch_pb2.py', 'python/generated/dispatch_pb2_grpc.py', 'python/offline_split.py',
              'grade_commitment_ppo_v1/features.py', 'single_critic_task_query_v1/c_features.py',
              'change_plans/token_attention_001/schema_audit_001/token_adapter_audit.py',
              'grade_event_memory_v1/targets.py', 'grade_event_memory_v1/events.py',
              'physical_curriculum_v1/targets.py', 'physical_realization_v1/engine.py',
              'relation_gat_ego_v1/adapter.py', 'baseline1_v1/collection.py',
              'business_graph_summary_campaign_v1/budget.py', 'business_graph_summary_campaign_v1/plan_io.py',
              'business_graph_summary_campaign_v1/launch.py', 'online_v1_1/operations.py')
    files = dict.fromkeys([Path(ref['path']) for ref in prior['source_pins']] + list(extra)
                          + [ROOT/name for name in shared] + [PLAN/name for name in names])
    return [reference(path) for path in files]


def job_for(slot, anchor):
    run, environment = FIRST_RUN+slot, anchor['experiment_seed']
    leaf = f'run_{run}'
    recording = f'{CAMPAIGN}-{slot:03d}-dispatch_fifo_v3'
    event_dir = EVENTS/leaf
    return dict(ordinal=slot, pair_ordinal=slot, policy_kind=POLICY, actor_seed=0,
                group='BASELINE', role='final', source_recording_id=recording, recording_id=recording,
                run_index=run, experiment_seed=environment, actor_sampling_seed=0,
                actor_sampling_seed_consumed=False, event_directory=str(event_dir),
                event_bundle=str(event_dir/episode_id(run, environment)), python_output=str(OUTPUT/'shifts'/leaf),
                target_training_update=0, checkpoint_update=0, scene_id=anchor['scene_id'],
                expected_rpc_config=anchor['expected_rpc_config'], runtime_scene_sha256=anchor['runtime_scene_sha256'],
                v2_reference_run_index=anchor['run_index'], v2_recording_id=anchor['recording_id'],
                v2_phase_config=anchor['phase_config'])


def phase_path(job):
    return PREPARED/'configs'/f"run_{job['run_index']}.json"


def config_for(job, protected_core, contract):
    cfg = smoke.config_for(job, {'protected_core':protected_core})
    cfg.update(schema=PHASE_SCHEMA, campaign_id=CAMPAIGN, method=METHOD, **contract)
    return cfg


def validate_jobs(jobs, anchors):
    require(len(jobs) == len(anchors) == COUNT, '100 paired FIFO DISPATCH jobs required')
    for key in ('run_index', 'recording_id', 'source_recording_id', 'event_directory', 'event_bundle',
                'python_output', 'experiment_seed', 'v2_reference_run_index'):
        require(len({row[key] for row in jobs}) == COUNT, 'duplicate paired identity '+key)
    for slot, (job, anchor) in enumerate(zip(jobs, anchors, strict=True)):
        bare = {key: val for key,val in job.items() if key != 'phase_config'}
        require(bare == job_for(slot,anchor) and type(job['experiment_seed']) is int
                and 0 < job['experiment_seed'] <= JAVA_MASK,
                'frozen V3 physical identity or matching V2 seed differs')


def load_suite(path=SUITE, *, check_sources=True):
    path = Path(path)
    require(path.resolve() == SUITE, 'frozen FIFO DISPATCH suite path required')
    suite = load_reference(reference(path))
    require(path.read_bytes() == encoded(suite), 'noncanonical FIFO DISPATCH suite bytes')
    require(suite['schema'] == SCHEMA and suite['campaign_id'] == CAMPAIGN and suite['execution'] == EXECUTION
            and suite['method'] == METHOD and all(suite.get(key) == val for key,val in LAYOUT.items())
            and suite['physical_shifts'] == COUNT and suite['evaluation_slots'] == SLOTS
            and suite['training_shifts'] == suite['optimizer_updates'] == 0 and suite['pairing'] == PAIRING,
            'FIFO DISPATCH suite scope or execution differs')
    formal, anchors = source_contract()
    require(suite['previous_formal_suite'] == reference(previous.SUITE)
            and suite['protected_core'] == formal['protected_core'], 'frozen V2 source or PPO bytes differ')
    contract = algorithm_contract()
    require(all(suite.get(key) == value for key,value in contract.items()), 'FIFO DISPATCH strategy contract differs')
    validate_jobs(suite['jobs'], anchors)
    for job in suite['jobs']:
        config = phase_path(job)
        require(reference(config) == job['phase_config']
                and config.read_bytes() == encoded(config_for(job,suite['protected_core'],contract)),
                'frozen FIFO DISPATCH phase bytes differ')
    for key, name in [('master_plan','master_plan.csv'), ('scenario_table','scenario_table.csv'),
                      ('java_helper','Baseline1CampaignPlan.java'), ('run_seed','run_seed.csv')]:
        require(reference(PREPARED/name) == suite[key], 'frozen FIFO DISPATCH launch artifact differs: '+key)
    if check_sources:
        require(suite['source_pins'] == source_pins(), 'FIFO DISPATCH source bytes differ')
    return suite


def load_v2_completion(suite):
    """Bind the old final report and all 100 independent sealed DISPATCH receipts."""
    import formal_ledger

    formal, anchors = source_contract()
    original = reference(previous.SUITE)
    require(suite['previous_formal_suite'] == original, 'V3 suite lost its frozen V2 source')
    path = previous.OUTPUT/'suite_complete.json'
    if not path.is_file():
        status_path = previous.OUTPUT/'run_status.json'
        if status_path.is_file():
            require(json.loads(status_path.read_bytes())['status'] != 'STOP',
                    'formal V2 stopped before the final 500-shift seal')
        raise PreviousV2Pending('formal V2 must finish and publish the complete 500-shift seal')
    completion_ref = reference(path)
    complete = load_reference(completion_ref)
    report_ref = reference(previous.REPORT/previous.REPORT_FILENAME)
    require(complete['status'] == 'COMPLETE' and complete['campaign_id'] == formal['campaign_id']
            and complete['physical_shifts'] == 500 and complete['evaluation_slots'] == SLOTS
            and complete['optimizer_updates'] == complete['training_shifts'] == 0
            and complete['protected_core'] == formal['protected_core']
            and complete['suite'] == original and complete['evaluation'] == report_ref,
            'formal V2 final seal differs from the pair protocol')
    report = load_reference(report_ref)
    require(report['schema'] == 'bg128-traditional-paired100-comparison-v2' and report['status'] == 'COMPLETE'
            and report['campaign_id'] == formal['campaign_id'] and report['pairing'] == formal['pairing']
            and report['method'] == formal['method'] and report['evaluation_slots'] == SLOTS
            and report['physical_shifts'] == report['completed_shifts'] == 500 and report['optimizer_updates'] == 0
            and len(report['shifts']) == len(report['receipts']) == 500, 'formal V2 final report differs')
    slots = formal_ledger.Ledger(formal).snapshot()['slots']
    fields = ('run_index','recording_id','experiment_seed','actor_sampling_seed','policy_kind','phase_config',
              'pair_ordinal','scene_id')
    for job, receipt, row in zip(formal['jobs'], report['receipts'], report['shifts'], strict=True):
        ledger_slot = slots[str(job['run_index'])]
        require(ledger_slot['identity'] == job and ledger_slot['status'] == 'COMPLETE'
                and ledger_slot['receipt'] == receipt
                and receipt['path'] == str(Path(job['python_output'])/'evaluation_receipt.json')
                and row == load_reference(receipt) and all(row[key] == job[key] for key in fields)
                and row['status'] == 'COMPLETE' and row['normal_completion'] is True
                and row['completed_shifts'] == 1 and row['optimizer_updates'] == 0,
                'all 500 formal V2 physical slots must be sealed and complete')
    validate_jobs(suite['jobs'], anchors)
    result = [dict(receipt=report['receipts'][job['ordinal']], row=report['shifts'][job['ordinal']]) for job in anchors]
    return dict(completion=completion_ref, report=report_ref, anchors=result)
