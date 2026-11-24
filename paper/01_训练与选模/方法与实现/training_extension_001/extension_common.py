"""Canonical scope for the BG128 u16-to-u32 continuation and paired evaluation."""
import csv
from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[3]
sys.path.insert(0, str(ROOT))

from baseline1_v1.java_plan import COLUMNS
from business_graph_summary_campaign_v1.gate import PROTECTED_SHA, check_phase
from business_graph_summary_campaign_v1.plan_io import load_phase, load_reference, reference, sha
from business_graph_summary_evaluation_v1.gate import require, seed
from business_graph_summary_evaluation_v1.gate import source_pins as inherited_pins

CAMPAIGN = 'bg128-u16-to-u32-20261119-001'
CANDIDATE = 's01_BG128'
OUTPUT = ROOT / 'runs' / CAMPAIGN
CONTROL = OUTPUT / 'control'
MATERIALS = OUTPUT.with_name(CAMPAIGN + '_materials')
ARM = OUTPUT / 'arms' / CANDIDATE
REPORT = ROOT / 'reports/token_attention_001/business_graph_summary_009/training_extension_001'
PREPARED = PLAN / 'prepared_001'
SUITE = PREPARED / 'suite.json'
APPROVAL = PREPARED / 'approval.json'
PARENT_PLAN = PLAN.parent / 'formal_training_001/prepared_001'
PARENT_OUTPUT = ROOT / 'runs/business-graph-summary-formal-seed1-20261118-001'
PARENT_CHECKPOINT = PARENT_OUTPUT / 'arms' / CANDIDATE / 'updates/0016/checkpoint.pt'
PARENT_CHECKPOINT_SHA = 'f0bf60a3091f85eb4f163806805f495ff70d0f7aa325906203c82c92ecc30e43'
SEED_SUITE = PLAN.parent / 'checkpoint_evaluation_001/prepared_001/suite.json'
PREDECESSOR = PLAN.parent / 'first_window_action_001/discovery_3e4_001/discovery_budget.py'
CORE = ROOT.parents[1] / 'python/episode_budget_ppo.py'
FIRST_RUN = 9990001
PREVIOUS_CAP = 21799
TRAIN_COUNT = 512
EVAL_COUNT = 300
TOTAL = TRAIN_COUNT + EVAL_COUNT
SCHEDULE = (
    [('eval', 16, 100)]
    + [('train', i, 32) for i in range(17, 25)]
    + [('eval', 24, 100)]
    + [('train', i, 32) for i in range(25, 33)]
    + [('eval', 32, 100)]
)
USER_INSTRUCTION = (
    '在当前16次更新延长到32次更新，同时在24和32次进行固定推理100班次，'
    '16次和32次训练的固定推理班次，优先使用之前 BG128 的同一套100个独立评价种子。先完成这次验证'
)


def checked(ref):
    path = Path(ref['path'])
    require(reference(path) == ref, 'frozen bytes changed: ' + str(path))
    return path


def source_pins():
    # The old collector and numerical code are pinned together with the thin
    # continuation, fixed-evaluation, budget and launch modules.
    pins = inherited_pins()
    for path in sorted(PLAN.glob('*.py')):
        if not path.name.startswith('test_'):
            pins[str(path)] = sha(path)
    pins[str(PREDECESSOR)] = sha(PREDECESSOR)
    return pins


def check_sources(suite):
    require(suite['source_pins'] == source_pins() and sha(CORE) == PROTECTED_SHA,
            'campaign sources or protected PPO core changed')


def training_seeds(ordinal):
    namespace = CAMPAIGN + '-s01-continuation'
    return tuple(seed(namespace, stream, ordinal) for stream in ('environment', 'sampling'))


def approval_for(suite):
    return dict(schema='bg128-training-extension-approval-v1', campaign_id=CAMPAIGN,
                suite_sha256=sha(SUITE), approved_by='user', decision='approved',
                source_user_instruction=USER_INSTRUCTION,
                approved_new_shifts=TOTAL, previous_approved_cap=PREVIOUS_CAP,
                new_approved_cap=PREVIOUS_CAP + TOTAL,
                new_training_shifts=TRAIN_COUNT, new_evaluation_shifts=EVAL_COUNT,
                first_training_update=17, last_training_update=32,
                paired_evaluation_updates=[16, 24, 32])


def load_package():
    import json

    suite = json.loads(SUITE.read_bytes())
    check_sources(suite)
    require(suite['schema'] == 'bg128-training-extension-v1' and suite['campaign_id'] == CAMPAIGN
            and suite['suite_path'] == str(SUITE) and suite['python_output'] == str(OUTPUT)
            and suite['control_directory'] == str(CONTROL) and suite['material_root'] == str(MATERIALS)
            and suite['budget_path'] == str(OUTPUT / 'extension_ledger.sqlite')
            and suite['new_shift_cap'] == TOTAL and suite['training_shifts'] == TRAIN_COUNT
            and suite['evaluation_shifts'] == EVAL_COUNT and suite['training_updates'] == 16
            and suite['cpu_threads'] == 2, 'campaign scope differs')
    require(suite['phase_configs'] == [item['phase_config'] for item in suite['phases']]
            and len(suite['phases']) == len(SCHEDULE), 'phase inventory differs')
    parent = load_reference(suite['training_suite'])
    completed = load_reference(suite['training_completion'])
    require(checked(suite['training_suite']) == PARENT_PLAN / 'suite.json'
            and completed['campaign'] == suite['training_suite']
            and completed['physical_shifts'] == 512 and completed['global_updates'] == 16
            and parent['arms'][0]['learner_config'] == suite['learner_config']
            and suite['learner_config']['ppo']['actor_lr'] == 3e-4
            and suite['arms'] == [dict(candidate_id=CANDIDATE, python_output=str(ARM),
                                      learner_config=suite['learner_config'])],
            'original 3e-4 learner configuration or completed lineage differs')
    original = suite['parent_artifacts']
    require(set(original) == {'checkpoint', 'manifest', 'seal'}
            and checked(original['checkpoint']) == PARENT_CHECKPOINT
            and original['checkpoint']['sha256'] == PARENT_CHECKPOINT_SHA
            and completed['checkpoints'] == [original['checkpoint']]
            and suite['parent_checkpoint'] == original['checkpoint'],
            'original u16 checkpoint identity differs')
    require(load_reference(original['manifest'])['update_index'] == 16,
            'original checkpoint manifest update differs')
    checked(original['seal'])
    historical = load_reference(suite['paired_seed_source'])
    require(checked(suite['paired_seed_source']) == SEED_SUITE and historical['checkpoint_update'] == 13
            and historical['learner_config'] == suite['learner_config'],
            'historical BG128 evaluation source differs')
    old_eval = load_phase(checked(historical['phase_config']))
    old_train = load_phase(checked(parent['phase_configs'][-1]))
    mutable = {'campaign_id', 'phase_id', 'arm_phase_id', 'arm_update', 'recording_id',
               'python_output', 'event_directory', 'jobs'}
    phases, jobs = [], []
    train_ordinal = 0
    for item, (role, update, size) in zip(suite['phases'], SCHEDULE):
        phase_path = checked(item['phase_config'])
        cfg = load_phase(phase_path)
        check_phase(cfg)
        short_name = f'{role}_u{update:04d}'
        phase_id = CANDIDATE + '_' + short_name
        recording = CAMPAIGN + '-' + phase_id
        directory = (ARM if role == 'train' else OUTPUT / 'evaluation') / short_name
        events = CONTROL / 'legacy_unused' / phase_id
        checkpoint = update - 1 if role == 'train' else update
        source_cfg = old_train if role == 'train' else old_eval
        require({k: v for k, v in cfg.items() if k not in mutable}
                == {k: v for k, v in source_cfg.items() if k not in mutable},
                'original RPC, material, graph, reward or recording contract differs')
        require(item == dict(phase_id=phase_id, role=role, checkpoint_update=checkpoint,
                             update_index=update, python_output=str(directory),
                             phase_config=reference(PREPARED / 'configs' / (phase_id + '.json'))),
                'phase schedule or output differs')
        require((cfg['campaign_id'], cfg['phase_id'], cfg['arm_phase_id'], cfg['arm_update'],
                 cfg['recording_id'], cfg['python_output'], cfg['event_directory'], len(cfg['jobs']))
                == (CAMPAIGN, phase_id, short_name, update, recording, str(directory), str(events), size),
                'phase configuration identity differs')
        for slot, job in enumerate(cfg['jobs']):
            ordinal = len(jobs)
            run = FIRST_RUN + ordinal
            source = source_cfg['jobs'][slot]
            if role == 'train':
                env, sampling = training_seeds(train_ordinal)
                require(job['pair_ordinal'] == train_ordinal, 'training seed ordinal differs')
                train_ordinal += 1
            else:
                env, sampling = source['experiment_seed'], source['actor_sampling_seed']
            leaf = f'dispatch-production-wait-shortfall-v2-run-{run}-seed-{env}'
            material = MATERIALS / phase_id
            require((job['ordinal'], job['run_index'], job['experiment_seed'], job['actor_sampling_seed'],
                     job['checkpoint_update'], job['target_training_update'], job['arm_update'], job['role'],
                     job['partition'], job['actor_seed'], job['candidate_id'], job['phase_id'],
                     job['arm_phase_id'], job['recording_id'], job['source_recording_id'],
                     job['python_output'], job['event_directory'], job['event_bundle'],
                     job['material_event_directory'], job['material_event_bundle'], job['material_journal'])
                    == (ordinal, run, env, sampling, checkpoint, update if role == 'train' else 0, update,
                        'train' if role == 'train' else 'final', 'train' if role == 'train' else 'validation',
                        suite['learner_config']['seed'], CANDIDATE, phase_id, short_name, recording, recording,
                        str(directory), str(events), str(events / leaf), str(material), str(material / leaf),
                        str(material / leaf / 'material_events.bin'))
                    and job['expected_rpc_config'] == source['expected_rpc_config']
                    and job['runtime_scene_sha256'] == source['runtime_scene_sha256'],
                    'canonical run, seed or RPC identity differs')
            jobs.append(job)
        phases.append(cfg)
    require(len(jobs) == TOTAL and train_ordinal == TRAIN_COUNT, 'experiment count differs')
    with checked(suite['master_plan']).open(newline='') as stream:
        require(list(csv.DictReader(stream)) == [{k: str(job[k]) for k in COLUMNS} for job in jobs],
                'Java master differs from frozen phase identities')
    checked(suite['java_helper'])
    require(json.loads(APPROVAL.read_bytes()) == approval_for(suite), 'authorization differs')
    require('jobs' not in suite, 'suite must use the canonical phase inventory')
    suite['jobs'] = jobs
    return suite, phases
