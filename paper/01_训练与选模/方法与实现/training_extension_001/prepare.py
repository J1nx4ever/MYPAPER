"""Prepare a frozen one-click continuation. This entry never starts or registers it."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile

from extension_common import (
    APPROVAL, ARM, CAMPAIGN, CANDIDATE, CONTROL, EVAL_COUNT, FIRST_RUN, MATERIALS, OUTPUT,
    PARENT_CHECKPOINT, PARENT_OUTPUT, PARENT_PLAN, PLAN, PREPARED, PREVIOUS_CAP, SCHEDULE,
    SEED_SUITE, SUITE, TOTAL, TRAIN_COUNT, approval_for, load_package, reference,
    require, source_pins, training_seeds,
)
from business_graph_summary_campaign_v1.plan_io import load_phase, write
from baseline1_v1.java_plan import _replace_once
from extension_budget import budget_snapshot, check_collisions


def read(path):
    return json.loads(path.read_bytes())


def java_helpers():
    spec = importlib.util.spec_from_file_location(
        'bg128_original_helper', PLAN.parent / 'formal_training_001/prepare_training.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.master_bytes, module.render_java


def render_paired_helper(suite, jobs, master, master_bytes, render_java):
    # The existing renderer validates unique seeds for ordinary unpaired jobs.
    # The ledger has already enforced the exact allowed pairing for all rows.
    # Render the first phase with that validator and bind Java to the full CSV.
    first = jobs[:100]
    with tempfile.TemporaryDirectory(prefix='bg128-extension-java-') as folder:
        projection = Path(folder) / 'first_phase.csv'
        projection.write_bytes(master_bytes(first))
        source = render_java(dict(suite, jobs=first), projection, suite_path=SUITE,
                             bridge_path=PLAN / 'bootstrap.py')
        for constant, old, new in (
            ('MASTER_PLAN_PATH', str(projection), str(master)),
            ('MASTER_PLAN_SHA256', reference(projection)['sha256'], reference(master)['sha256']),
        ):
            prefix = f'private static final String {constant} = '
            source = _replace_once(source, prefix + json.dumps(old) + ';',
                                   prefix + json.dumps(new) + ';')
    return _replace_once(source, 'private static final int JOB_COUNT = 100;',
                         f'private static final int JOB_COUNT = {TOTAL};')


def build_jobs(parent, historical):
    frozen_train = read(Path(parent['phase_configs'][-1]['path']))
    frozen_eval = load_phase(historical['phase_config']['path'])
    all_jobs, phase_items, phase_documents, identities = [], [], [], []
    train_ordinal = 0
    for role, update, count in SCHEDULE:
        short_name = f'{role}_u{update:04d}'
        phase_id = CANDIDATE + '_' + short_name
        recording = CAMPAIGN + '-' + phase_id
        directory = (ARM if role == 'train' else OUTPUT / 'evaluation') / short_name
        events = CONTROL / 'legacy_unused' / phase_id
        materials = MATERIALS / phase_id
        checkpoint = update - 1 if role == 'train' else update
        source = frozen_train if role == 'train' else frozen_eval
        defaults = (read(Path(source['common_reference']['path']))['job_defaults']
                    if role == 'train' else {})
        cfg = dict(source, campaign_id=CAMPAIGN, phase_id=phase_id, arm_phase_id=short_name,
                   arm_update=update, recording_id=recording, python_output=str(directory),
                   event_directory=str(events), jobs=[])
        for slot in range(count):
            ordinal = len(all_jobs)
            run = FIRST_RUN + ordinal
            old = source['jobs'][slot]
            if role == 'train':
                env, sampling = training_seeds(train_ordinal)
                group = CAMPAIGN + f'-s01-train-{train_ordinal:04d}'
            else:
                env, sampling = old['experiment_seed'], old['actor_sampling_seed']
                group = old['realization_group_id']
            leaf = f'dispatch-production-wait-shortfall-v2-run-{run}-seed-{env}'
            job = dict(old, ordinal=ordinal, run_index=run, experiment_seed=env, actor_sampling_seed=sampling,
                       arm_update=update, arm_phase_id=short_name, phase_id=phase_id,
                       target_training_update=update if role == 'train' else 0,
                       checkpoint_update=checkpoint, recording_id=recording, source_recording_id=recording,
                       realization_group_id=group, scene_group=group, python_output=str(directory),
                       event_directory=str(events), event_bundle=str(events / leaf),
                       material_event_directory=str(materials), material_event_bundle=str(materials / leaf),
                       material_journal=str(materials / leaf / 'material_events.bin'))
            if role == 'train':
                job['pair_ordinal'] = train_ordinal
                train_ordinal += 1
            cfg['jobs'].append(job)
            all_jobs.append(dict(defaults, **job))
        path = PREPARED / 'configs' / (phase_id + '.json')
        phase_documents.append((path, cfg))
        phase_items.append(dict(phase_id=phase_id, role=role, checkpoint_update=checkpoint,
                                update_index=update, python_output=str(directory)))
        identities.append(dict(role=role, update=update, checkpoint_update=checkpoint,
                               recording_id=recording, python_output=str(directory),
                               anylogic_event_directory=str(materials), first_run=cfg['jobs'][0]['run_index'],
                               last_run=cfg['jobs'][-1]['run_index'], shifts=count))
    return all_jobs, phase_items, phase_documents, identities


def prepare():
    require(not PREPARED.exists() and not OUTPUT.exists() and not MATERIALS.exists(),
            'preparation and experiment identities must be new')
    historical = read(SEED_SUITE)
    parent = read(PARENT_PLAN / 'suite.json')
    jobs, phases, documents, identities = build_jobs(parent, historical)
    config = parent['arms'][0]['learner_config']
    artifacts = dict(checkpoint=reference(PARENT_CHECKPOINT),
                     manifest=reference(str(PARENT_CHECKPOINT) + '.manifest.json'),
                     seal=reference(str(PARENT_CHECKPOINT) + '.seal.json'))
    suite = dict(
        schema='bg128-training-extension-v1', campaign_id=CAMPAIGN, suite_path=str(SUITE),
        new_shift_cap=TOTAL, training_shifts=TRAIN_COUNT, evaluation_shifts=EVAL_COUNT,
        training_updates=16, first_update=17, final_update=32, fixed_evaluation_updates=[16, 24, 32],
        cpu_threads=2, training_seed_count=1, episodes_per_batch=32, stage='formal',
        execution_profile=config['execution_profile'], pair_id=CAMPAIGN + '-paired-u16-u24-u32',
        run_index_base=FIRST_RUN, actor_seed=config['seed'], sampling='stochastic_private_rng',
        python_output=str(OUTPUT), control_directory=str(CONTROL), material_root=str(MATERIALS),
        budget_path=str(OUTPUT / 'extension_ledger.sqlite'), occupied_before=PREVIOUS_CAP,
        maximum_total_if_approved=PREVIOUS_CAP + TOTAL, active_h2_reserved_shifts=0,
        phases=phases, learner_config=config, parent_artifacts=artifacts,
        parent_checkpoint=artifacts['checkpoint'],
        arms=[dict(candidate_id=CANDIDATE, python_output=str(ARM), learner_config=config)],
        paired_seed_source=reference(SEED_SUITE), training_suite=reference(PARENT_PLAN / 'suite.json'),
        training_completion=reference(PARENT_OUTPUT / 'suite_complete.json'),
        one_click_entry=reference(PLAN / 'bootstrap.py'), source_pins=source_pins(),
        checkpoint_binding='original sealed u16; u24/u32 exact committed updates of this extension',
    )
    previous = budget_snapshot()
    require(previous['maximum_total_shifts'] == PREVIOUS_CAP, 'merged predecessor budget changed')
    for item, identity, (path, cfg) in zip(phases, identities, documents):
        write(path, cfg)
        item['phase_config'] = identity['config'] = reference(path)
    suite['phase_configs'] = [item['phase_config'] for item in phases]
    check_collisions(dict(suite, jobs=jobs), previous)
    master_bytes, render_java = java_helpers()
    master = PREPARED / 'master_plan.csv'
    master.write_bytes(master_bytes(jobs))
    suite['master_plan'] = reference(master)
    helper = PREPARED / 'Baseline1CampaignPlan.java'
    helper.write_text(render_paired_helper(suite, jobs, master, master_bytes, render_java),
                      encoding='utf-8')
    suite['java_helper'] = reference(helper)
    write(SUITE, suite)
    write(APPROVAL, approval_for(suite))
    write(PREPARED / 'launch_identity.json',
          dict(suite=reference(SUITE), parent=artifacts['checkpoint'], physical_shifts=TOTAL,
               source_paired_seed_suite=suite['paired_seed_source'], phases=identities))
    columns = ('ordinal', 'role', 'arm_update', 'checkpoint_update', 'recording_id', 'config_sha256',
               'python_output', 'material_event_directory', 'run_index', 'experiment_seed',
               'actor_sampling_seed', 'actor_seed')
    digests = {item['recording_id']: item['config']['sha256'] for item in identities}
    with (PREPARED / 'identities.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: row[k] for k in columns}
                         for row in (dict(job, config_sha256=digests[job['recording_id']]) for job in jobs))
    load_package()
    print(json.dumps(dict(suite=reference(SUITE), shifts=TOTAL, status='PREPARED_ONLY')))


if __name__ == '__main__':
    prepare()
