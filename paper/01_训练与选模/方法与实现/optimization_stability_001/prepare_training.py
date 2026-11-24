"""Freeze BG128 16x32 online training identities; never register or start a run."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

ROOT = Path('/home/jinx/Faith/experiments/lagrangian_ppo_v1')
sys.path.insert(0, str(ROOT))

CAMPAIGN = 'business-graph-stability-lr1e4-seed1-20261118-001'
RUN_BASE = 9960001
CANDIDATE = 's01_BG128'


def master_bytes(jobs):
    from baseline1_v1.java_plan import COLUMNS
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator='\n')
    writer.writeheader()
    writer.writerows({key: job[key] for key in COLUMNS} for job in jobs)
    return stream.getvalue().encode()


def render_java(suite, master, *, suite_path, bridge_path):
    """Reuse the audited immutable master/ready/claim protocol without mutation."""
    from baseline1_v1.java_plan import COLUMNS, render_java as render_base, _replace_once
    from baseline1_v1.prepare import _reference_configs
    from python.native_behavior import identity
    from single_critic_task_query_v1.java_plan import _with_startup
    from business_graph_stability_campaign_v1.plan_io import sha
    digests = {'P' + str(key): identity(value['behavior'])
               for key, value in _reference_configs().items()}
    source = render_base(master_path=str(master), master_sha256=sha(master),
        control_directory=suite['control_directory'], campaign_id=suite['campaign_id'],
        jobs=[{key: job[key] for key in COLUMNS} for job in suite['jobs']],
        behavior_digests=digests)
    material = str(Path(suite['python_output']).with_name(suite['campaign_id'] + '_materials'))
    extra = '''
    public static boolean materialEnabled(int index) { row(index); return true; }
    public static String phaseId(int index) {
        String recording = row(index).cells[5];
        require(recording.startsWith(CAMPAIGN_ID + "-"), "BG128 recording prefix differs");
        return recording.substring(CAMPAIGN_ID.length() + 1);
    }
    public static String materialRoot(int index) { return MATERIAL_ROOT + "/" + phaseId(index); }
    public static String stageId(int index) { return phaseId(index).split("_", 3)[1]; }
    /** Metadata compatibility only. Python uses the fixed combined cost. */
    public static double waitWeight(int index) { row(index); return 0.2; }
    public static long actorSamplingSeed(int index) { return positive(row(index).cells[8], "actor sampling seed"); }
    public static boolean preparationOnly() { return false; }
    public static int jobCount() { return JOB_COUNT; }
    public static int resumeOffset() { return 0; }
    public static int frozenPlanJobCount() { return JOB_COUNT; }
'''.replace('MATERIAL_ROOT', json.dumps(material))
    anchor = '    public static String eventRoot(int index) { return row(index).cells[9]; }'
    source = _replace_once(source, anchor, anchor + extra)
    return _with_startup(source, suite_path=suite_path, bridge_path=bridge_path)


def prepare(*, acceptance, directory=None):
    from business_graph_stability_campaign_v1.contract import seed, require
    from business_graph_stability_campaign_v1.plan_io import PLAN, reference, write
    from business_graph_stability_campaign_v1.gate import (
        CAMPAIGN_CONTRACT, PHASE_CONTRACT, TRAINING_REWARD, DIAGNOSTIC_SCOPE,
        source_pins, check_jobs, evaluation_contract)
    from business_graph_stability_campaign_v1.budget import (
        budget_snapshot, preceding_reference, check_collisions)
    from business_graph_stability_v1.contract import ablation_contract, feature_config_for
    from business_graph_stability_v1.learner import default_config
    from business_graph_stability_v1.c_features import feature_contract
    from business_graph_stability_v1.targets import composite_contract, REWARD_VERSION
    directory = Path(directory or PLAN / 'prepared_001').resolve()
    require(directory.parent == PLAN and not directory.exists(),
            'new immutable preparation directory under this plan required')
    accepted = json.loads(Path(acceptance).read_bytes())
    require(accepted.get('status') == 'PASS_OFFLINE_ONLY' and
            accepted.get('source_pins') == source_pins(),
            'current offline acceptance required before freezing a package')
    output = ROOT / 'runs' / CAMPAIGN
    materials = output.with_name(CAMPAIGN + '_materials')
    require(not output.exists() and not materials.exists(), 'fresh run identity occupied')
    before = budget_snapshot()
    state = preceding_reference()
    source_common = ROOT / 'change_plans/token_attention_001/single_critic_task_query_008/engineering_001/formal_prepared_002/common.json'
    common = json.loads(source_common.read_bytes())
    common.update(contract=PHASE_CONTRACT, execution_profile='formal_32ep_v1',
        composite_contract=composite_contract(), training_reward=TRAINING_REWARD,
        diagnostic_scope=DIAGNOSTIC_SCOPE)
    common['material_memory']['reward_version'] = REWARD_VERSION
    common['diagnostic_recording'] = dict(transition_jsonl=False, rpc_jsonl=False,
                                          rollout_snapshots=False)
    write(directory / 'common.json', common)
    common_ref = reference(directory / 'common.json')
    from business_graph_stability_campaign_v1.pairing import training_source, contract as pairing_contract
    paired_source = training_source()
    init = paired_source['arms'][0]['actor_seed']
    arms = [dict(candidate_id=CANDIDATE, representation='BG128', replicate='s01', actor_seed=init,
        learner_config=default_config(seed=init, candidate_id=CANDIDATE,
                                      execution_profile='formal_32ep_v1'),
        python_output=str(output / 'arms' / CANDIDATE), new_shift_cap=512)]
    jobs, phases, identities = [], [], []
    for local in range(1, 17):
        name = f'train_u{local:04d}'
        phase = CANDIDATE + '_' + name
        recording = CAMPAIGN + '-' + phase
        events = output / 'control/legacy_unused' / phase
        material = materials / phase
        cfg = dict(common_reference=common_ref, campaign_id=CAMPAIGN, phase_id=phase,
            arm_phase_id=name, arm_update=local, candidate_id=CANDIDATE, recording_id=recording,
            python_output=str(output / 'arms' / CANDIDATE / name), event_directory=str(events),
            feature_contract=feature_contract(feature_config_for(CANDIDATE)),
            graph_contract=ablation_contract(CANDIDATE), jobs=[])
        for inside in range(32):
            ordinal = len(jobs)
            run = RUN_BASE + ordinal
            namespace = CAMPAIGN + '-s01-train'
            env = paired_source['jobs'][ordinal]['experiment_seed']
            sampling = paired_source['jobs'][ordinal]['actor_sampling_seed']
            episode = f'dispatch-production-wait-shortfall-v2-run-{run}-seed-{env}'
            job = dict(ordinal=ordinal, pair_ordinal=ordinal, candidate_id=CANDIDATE,
                arm_phase_id=name, arm_update=local, actor_seed=init, role='train',
                target_training_update=local, checkpoint_update=local - 1,
                run_index=run, experiment_seed=env, actor_sampling_seed=sampling,
                scene_group=namespace + f'-pair-{ordinal:04d}',
                realization_group_id=namespace + f'-pair-{ordinal:04d}', partition='train',
                phase_id=phase, recording_id=recording, source_recording_id=recording,
                event_directory=str(events), event_bundle=str(events / episode),
                python_output=cfg['python_output'], material_event_directory=str(material),
                material_event_bundle=str(material / episode),
                material_journal=str(material / episode / 'material_events.bin'))
            cfg['jobs'].append(job)
            jobs.append(dict(common['job_defaults'], **job))
        path = directory / 'configs' / (phase + '.json')
        write(path, cfg)
        ref = reference(path)
        phases.append(ref)
        identities.append(dict(recording_id=recording, config=ref, candidate_id=CANDIDATE,
            role='train', python_output=cfg['python_output'], anylogic_event_directory=str(material),
            jobs=[{key: job[key] for key in ('run_index', 'experiment_seed',
                  'actor_sampling_seed', 'actor_seed', 'pair_ordinal')} for job in cfg['jobs']]))
    suite = dict(contract=CAMPAIGN_CONTRACT, stage='formal', campaign_id=CAMPAIGN,
        optimization_protocol=reference(PLAN/'protocol.json'),
        pairing=pairing_contract(),
        kind='single_seed_trial', suite_path=str(directory / 'suite.json'),
        execution_profile='formal_32ep_v1', pair_id=CAMPAIGN + '-single',
        new_shift_cap=512, training_shifts=512, evaluation_shifts=0, training_updates=16,
        training_seed_count=1, episodes_per_batch=32, arm_order=[CANDIDATE], arms=arms,
        python_output=str(output), control_directory=str(output / 'control'),
        run_index_base=RUN_BASE, phase_configs=phases, common_config=common_ref, cpu_threads=2,
        source_pins=source_pins(), engineering_acceptance=reference(acceptance),
        budget_state_before=state, budget_active_scope_total_before=before['maximum_total_shifts'],
        pending_predecessor_shifts=0, legacy_reserved_shifts=0, active_h2_reserved_shifts=0,
        maximum_total_if_approved=before['maximum_total_shifts'] + 512,
        approval_status='PREPARED_USER_RUN_CLICK_REQUIRED', jobs=jobs, predecessor_suites=[],
        one_click_entry=reference(PLAN / 'bootstrap.py'), evaluation_contract=evaluation_contract(False),
        initialization='original_common_initial_checkpoint_zero_updates',
        formal_model_inherits_smoke_weights=False, formal_campaign_auto_start=False,
        first_batch_acceptance='first_32_formal_shifts_full_RPC_collection_PPO_commit_then_continue',
        decision_diagnostics='disabled_high_frequency_diagnostics')
    check_jobs(suite)
    check_collisions(suite, before)
    master = directory / 'master_plan.csv'
    master.write_bytes(master_bytes(jobs))
    suite['master_plan'] = reference(master)
    java = directory / 'Baseline1CampaignPlan.java'
    java.write_text(render_java(suite, master, suite_path=directory / 'suite.json',
                                bridge_path=PLAN / 'bootstrap.py'))
    suite['java_helper'] = reference(java)
    disk = dict(suite)
    disk.pop('jobs')
    write(directory / 'suite.json', disk)
    write(directory / 'launch_identity.json', dict(suite=reference(directory / 'suite.json'),
        stage='formal', physical_shifts=512, status='PREPARED_NOT_REGISTERED_NOT_STARTED',
        configurations=identities))
    with (directory / 'identities.csv').open('x', newline='') as stream:
        columns = ('recording_id', 'config_sha256', 'python_output', 'anylogic_event_directory',
                   'run_index', 'experiment_seed', 'actor_sampling_seed', 'actor_seed')
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        for item in identities:
            for job in item['jobs']:
                writer.writerow(dict(recording_id=item['recording_id'],
                    config_sha256=item['config']['sha256'], python_output=item['python_output'],
                    anylogic_event_directory=item['anylogic_event_directory'],
                    **{key: job[key] for key in columns[4:]}))
    return reference(directory / 'suite.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acceptance', required=True)
    parser.add_argument('--directory')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Explicit preparation mode; this program never launches or registers.')
    args = parser.parse_args()
    print(json.dumps(prepare(acceptance=args.acceptance, directory=args.directory), ensure_ascii=False))


if __name__ == '__main__':
    main()
