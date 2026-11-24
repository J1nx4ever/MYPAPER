"""Freeze fresh continuous-grade BG128 training and paired evaluation identities."""

from collections import Counter
import csv
from itertools import permutations
import json
from pathlib import Path
from random import Random
import sys

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline1_v1.java_plan import COLUMNS
from business_graph_summary_campaign_v1.contract import require, seed
from business_graph_summary_campaign_v1.gate import PROTECTED_SHA, check_phase
from business_graph_summary_campaign_v1.plan_io import load_phase, load_suite, reference, sha, write
from business_graph_summary_evaluation_v1.gate import source_pins as inherited_pins
from python.runtime_scenario import PREFIX, fingerprint

CAMPAIGN = 'bg128-continuous-mixed-20261119-001'
CANDIDATE = 's01_BG128'
OUTPUT = ROOT / 'runs' / CAMPAIGN
CONTROL = OUTPUT / 'control'
MATERIALS = OUTPUT.with_name(CAMPAIGN + '_materials')
LAUNCH = OUTPUT / 'launch'
PREPARED = PLAN / 'prepared_001'
SUITE = PREPARED / 'suite.json'
APPROVAL = LAUNCH / 'run_authority.json'
LEDGER = OUTPUT / 'budget.sqlite'
CORE = ROOT.parents[1] / 'python/episode_budget_ppo.py'
REPORT = ROOT / 'reports/token_attention_001/business_graph_summary_009/generalization_001/continuous_training_001'
GENERALIZATION = PLAN.parents[1]
EXTENSION = GENERALIZATION.parent / 'training_extension_001'
FORMAL_TRAINING = GENERALIZATION.parent / 'formal_training_001'
TRAIN_TEMPLATE = FORMAL_TRAINING / 'prepared_001/configs/s01_BG128_train_u0001.json'
EVAL_TEMPLATE = EXTENSION / 'prepared_001/configs/s01_BG128_eval_u0032.json'
OLD_CHECKPOINT = ROOT / 'runs/bg128-u16-to-u32-20261119-001/arms/s01_BG128/updates/0032/checkpoint.pt'
OLD_CHECKPOINT_SHA = '5d42df7d8e33b5f11424094268a4da894a36454e46d86889fc0218ecce6b2914'
FIRST_RUN = 10000001
PREVIOUS_CAP = 23211
BATCH_SIZE = 32
UPDATES = 32
EVAL_COUNT = 100
TRAINING_SHIFTS = BATCH_SIZE * UPDATES
MEAN_RANGES = {'H': (.36, .38), 'M': (.29, .30), 'L': (.15, .17)}
HALF_WIDTH_RANGE = (.005, .01)
ROLES = ('H', 'M', 'L', 'W')
ANCHOR = (.38, .30, .15)
EVAL_CASES = {
    'nominal': ('HMLW', ANCHOR, .005),
    'grade_swap': ('LMHW', ANCHOR, .005),
    'held_order_1': ('HLMW', ANCHOR, .005),
    'held_order_2': ('MHLW', ANCHOR, .005),
    'new_numeric': ('HMLW', (.37, .295, .16), .0075),
    'joint_shift': ('LHWM', (.365, .293, .166), .0087),
    'waste_swap': ('HMWL', ANCHOR, .005),
}
HOLDOUTS = tuple(order for order, _, _ in EVAL_CASES.values() if order != 'HMLW')
EVAL_SHIFTS = 2 * len(EVAL_CASES) * EVAL_COUNT
TOTAL = TRAINING_SHIFTS + EVAL_SHIFTS
MAX_JAVA_SEED = (1 << 48) - 1
SUITE_SCHEMA = 'bg128-continuous-mixed-v1'
AUTHORITY_SCHEMA = 'bg128-continuous-mixed-launch-authority-v1'
IDENTITY_COLUMNS = ('ordinal', 'policy_arm', 'scene_id', 'assignment', 'arm_update', 'checkpoint_update',
                    'pair_ordinal', 'recording_id', 'config_sha256', 'runtime_scene_sha256', 'python_output',
                    'material_event_directory', 'run_index', 'experiment_seed', 'actor_sampling_seed', 'actor_seed')


def read(path):
    return json.loads(Path(path).read_bytes())


def source_pins():
    pins = inherited_pins()
    extra = [*PLAN.glob('*.py'), *PLAN.glob('Main.*.java'),
             GENERALIZATION / 'gen_budget.py', EXTENSION / 'train_extension.py']
    pins.update({str(path): sha(path) for path in extra})
    return dict(sorted(pins.items()))


def _check_artifacts():
    checkpoint = reference(OLD_CHECKPOINT)
    manifest = reference(str(OLD_CHECKPOINT) + '.manifest.json')
    seal = reference(str(OLD_CHECKPOINT) + '.seal.json')
    meta = read(manifest['path'])
    require(checkpoint['sha256'] == OLD_CHECKPOINT_SHA
            and meta['sha256'] == OLD_CHECKPOINT_SHA
            and meta['update_index'] == UPDATES
            and read(seal['path']) == {'sha256': OLD_CHECKPOINT_SHA, 'manifest_sha256': manifest['sha256']},
            'historical u32 checkpoint identity differs')
    require(sha(CORE) == PROTECTED_SHA, 'protected PPO core byte SHA changed')
    return dict(checkpoint=checkpoint, manifest=manifest, seal=seal)


def _phase_specs():
    for update in range(1, UPDATES + 1):
        yield 'train', 'mixed', update, BATCH_SIZE
    for scenario in EVAL_CASES:
        for policy_arm in ('old', 'new'):
            yield policy_arm, scenario, UPDATES, EVAL_COUNT


def _training_scenes(update):
    """Randomize continuous priors with eight episodes per possible waste site."""
    generator = Random(seed(CAMPAIGN, 'configuration', update))
    orders = [''.join(order) for order in permutations(ROLES) if ''.join(order) not in HOLDOUTS]
    scenes = []
    for waste in range(4):
        eligible = [order for order in orders if order[waste] == 'W']
        for _ in range(BATCH_SIZE // 4):
            means = tuple(generator.uniform(*bounds) for bounds in MEAN_RANGES.values())
            half_width = generator.uniform(*HALF_WIDTH_RANGE)
            scenes.append((generator.choice(eligible), means, half_width))
    generator.shuffle(scenes)
    return scenes


def _runtime_config(template, scene):
    order, means, half_width = scene
    grades = dict(zip(ROLES, (*means, 0.0), strict=True))
    require(sorted(order) == sorted(ROLES) and HALF_WIDTH_RANGE[0] <= half_width <= HALF_WIDTH_RANGE[1]
            and all(lo <= val <= hi for val, (lo, hi) in zip(means, MEAN_RANGES.values(), strict=True))
            and min(means) < .26 <= .28 < max(means), 'invalid or nonblendable mixed grade configuration')
    result = dict(template)
    for site, role in enumerate(order):
        prefix = f'{PREFIX}load.{site}.'
        result[prefix + 'isOre'] = float(role != 'W')
        result[prefix + 'grade'] = grades[role]
        if role != 'W':
            result[prefix + 'gradeHalfRange'] = half_width
            require(half_width <= grades[role] <= 1 - half_width, 'grade support leaves physical bounds')
        else:
            # Preserve the existing waste placeholder, which is unused for ore blending.
            result[prefix + 'gradeHalfRange'] = .005
    return result, fingerprint({key: value for key, value in result.items() if key.startswith(PREFIX)})


def _phase_documents(train_seed, old_seed):
    train_template = load_phase(TRAIN_TEMPLATE)
    eval_template = load_phase(EVAL_TEMPLATE)
    documents, jobs = [], []
    for policy_arm, scenario, update, count in _phase_specs():
        training = policy_arm == 'train'
        name = f'train_u{update:04d}' if training else f'eval_{scenario}_{policy_arm}'
        phase = f'{CANDIDATE}_{name}'
        recording = f'{CAMPAIGN}-{phase}'
        output = OUTPUT / 'arms' / CANDIDATE / name if training else OUTPUT / 'evaluation' / scenario / policy_arm
        events = CONTROL / 'legacy_unused' / phase
        materials = MATERIALS / phase
        template = train_template if training else eval_template
        cfg = dict(template, campaign_id=CAMPAIGN, phase_id=phase, arm_phase_id=name, arm_update=update,
                   recording_id=recording, python_output=str(output), event_directory=str(events),
                   policy_arm=policy_arm, scenario_id=scenario, jobs=[])
        scenes = _training_scenes(update) if training else [EVAL_CASES[scenario]] * count
        for slot, scene in enumerate(scenes):
            ordinal = len(jobs)
            pair = ordinal if training else slot
            run_index = FIRST_RUN + ordinal
            namespace = f'{CAMPAIGN}-train' if training else f'{CAMPAIGN}-evaluation'
            environment = seed(namespace, 'environment', pair)
            sampling = seed(namespace, 'sampling', pair)
            leaf = f'dispatch-production-wait-shortfall-v2-run-{run_index}-seed-{environment}'
            source = template['jobs'][slot]
            expected, scene_sha = _runtime_config(source['expected_rpc_config'], scene)
            scene_id = f'mixed_{scene[0].lower()}' if training else scenario
            group = f'{CAMPAIGN}-{scene_id}-{pair:04d}'
            job = dict(source, ordinal=ordinal, pair_ordinal=pair, run_index=run_index,
                       experiment_seed=environment, actor_sampling_seed=sampling,
                       actor_seed=old_seed if policy_arm == 'old' else train_seed,
                       arm_update=update, arm_phase_id=name, phase_id=phase,
                       target_training_update=update if training else 0,
                       checkpoint_update=update - 1 if training else update,
                       role='train' if training else 'final', partition='train' if training else 'validation',
                       recording_id=recording, source_recording_id=recording, scene_id=scene_id,
                       scene_group=group, realization_group_id=group, runtime_scene_sha256=scene_sha,
                       expected_rpc_config=expected, python_output=str(output),
                       event_directory=str(events), event_bundle=str(events / leaf),
                       material_event_directory=str(materials), material_event_bundle=str(materials / leaf),
                       material_journal=str(materials / leaf / 'material_events.bin'))
            cfg['jobs'].append(job)
            jobs.append(job)
        require(len(cfg['jobs']) == count, 'phase scenario count differs')
        documents.append(cfg)
    return documents, jobs


def _seed_pairs(rows):
    return [(row['experiment_seed'], row['actor_sampling_seed']) for row in rows]


def _check_inventory(suite, phases):
    require(len(phases) == UPDATES + 2 * len(EVAL_CASES), 'training/evaluation phase inventory differs')
    train_seed = suite['arms'][0]['actor_seed']
    old_seed = suite['old_learner_config']['seed']
    expected_phases, expected_jobs = _phase_documents(train_seed, old_seed)
    require(phases == expected_phases and suite['jobs'] == expected_jobs,
            'frozen expanded phase configurations or job inventory differ')
    for cfg in phases:
        check_phase(cfg)
    jobs = suite['jobs']
    require(len(jobs) == TOTAL and len({job['run_index'] for job in jobs}) == TOTAL
            and len({job['material_event_bundle'] for job in jobs}) == TOTAL,
            'mixed physical run identities are incomplete or duplicated')
    training, evaluation = jobs[:TRAINING_SHIFTS], jobs[TRAINING_SHIFTS:]
    require(_seed_pairs(evaluation) == _seed_pairs(evaluation[:EVAL_COUNT]) * (2 * len(EVAL_CASES)),
            'evaluation seed reuse must be limited to exact paired slots')
    train_scenes = {job['runtime_scene_sha256'] for job in training}
    eval_scenes = {job['runtime_scene_sha256'] for job in evaluation}
    require(len(train_scenes) == TRAINING_SHIFTS and len(eval_scenes) == len(EVAL_CASES)
            and train_scenes.isdisjoint(eval_scenes),
            'continuous training or held-out numeric scenario fingerprints overlap')
    seeds = [seed for pair in _seed_pairs(training + evaluation[:EVAL_COUNT]) for seed in pair]
    require(len(set(seeds + [train_seed, old_seed])) == len(seeds) + 2
            and all(0 < value <= MAX_JAVA_SEED for value in seeds + [train_seed, old_seed]),
            'new training/evaluation seed equality, range, or cross-type collision')
    held = set(HOLDOUTS)
    observed_roles = [set() for _ in range(4)]
    for update in range(1, UPDATES + 1):
        orders = [order for order, _, _ in _training_scenes(update)]
        require(not held.intersection(orders)
                and Counter(order.index('W') for order in orders) == Counter({site: 8 for site in range(4)}),
                'train permutations overlap evaluation holdouts or waste balance differs')
        for order in orders:
            for site, role in enumerate(order):
                observed_roles[site].add(role)
    require(all(roles == set(ROLES) for roles in observed_roles), 'every site must see every material/grade role')


def predecessor_snapshot():
    from change_plans.token_attention_001.business_graph_summary_009.generalization_001.gen_budget import budget_snapshot

    return budget_snapshot()


def check_collisions(suite, snapshot=None):
    from physical_curriculum_v1.budget import _history_rows
    from python.offline_split import java_random_states

    previous = predecessor_snapshot() if snapshot is None else snapshot
    require(previous['maximum_total_shifts'] == PREVIOUS_CAP
            and previous['active_h2_reserved_shifts'] == 0 and CAMPAIGN not in previous['campaigns'],
            'predecessor registered budget changed')
    jobs = suite['jobs']
    history = _history_rows(previous)
    for key in ('run_index', 'recording_id', 'source_recording_id'):
        existing = {job.get(key) for job in history}
        require(all(job[key] not in existing for job in jobs), 'historical identity collision: ' + key)
    independent_rows = jobs[:TRAINING_SHIFTS + EVAL_COUNT]
    new_seeds = [value for pair in _seed_pairs(independent_rows) for value in pair]
    new_seeds.append(suite['arms'][0]['actor_seed'])
    old_seeds = {job[key] & MAX_JAVA_SEED for job in history
                 for key in ('experiment_seed', 'actor_sampling_seed', 'actor_seed') if type(job.get(key)) is int}
    require(len(new_seeds) == len(set(new_seeds)) and old_seeds.isdisjoint(new_seeds),
            'new run uses a reserved or historical seed')
    historical_java = {state for job in history if type(job.get('experiment_seed')) is int
                       for state in java_random_states(job['experiment_seed']).values()}
    new_java = [state for job in independent_rows
                for state in java_random_states(job['experiment_seed']).values()]
    require(len(new_java) == len(set(new_java)) and historical_java.isdisjoint(new_java),
            'new run uses an existing Java engine or geological random state')
    return previous


def authorization_for(suite):
    require(suite['campaign_id'] == CAMPAIGN and Path(suite['suite_path']) == SUITE,
            'launch authority requires the canonical frozen package')
    return dict(schema=AUTHORITY_SCHEMA, campaign_id=CAMPAIGN, suite_sha256=sha(SUITE),
                max_new_shifts=TOTAL, training_shifts=TRAINING_SHIFTS, evaluation_shifts=EVAL_SHIFTS,
                training_updates=UPDATES, optimizer_updates=UPDATES, authorized_by_anylogic_run=True,
                previous_approved_cap=PREVIOUS_CAP, new_approved_cap=PREVIOUS_CAP + TOTAL,
                old_checkpoint_sha256=OLD_CHECKPOINT_SHA, paired_slots=EVAL_COUNT,
                scenarios=list(EVAL_CASES),
                source_user_instruction='保持原模型，连续抽取矿源品位及波动，并允许固定位置承担不同矿废角色。')


def _suite_settings(train_config, old_config, artifacts):
    return dict(schema=SUITE_SCHEMA, campaign_id=CAMPAIGN, suite_path=str(SUITE),
                candidate_id=CANDIDATE, arm_order=[CANDIDATE], stage='formal',
                execution_profile=train_config['execution_profile'], learner_config=train_config,
                arms=[dict(candidate_id=CANDIDATE, learner_config=train_config, actor_seed=train_config['seed'],
                           python_output=str(OUTPUT / 'arms' / CANDIDATE), new_shift_cap=TRAINING_SHIFTS)],
                old_checkpoint=artifacts['checkpoint'], old_artifacts=artifacts, old_learner_config=old_config,
                training_shifts=TRAINING_SHIFTS, evaluation_shifts=EVAL_SHIFTS, new_shift_cap=TOTAL,
                training_updates=UPDATES, first_update=1, final_update=UPDATES, episodes_per_batch=BATCH_SIZE,
                actor_seed=train_config['seed'], training_seed_count=1, cpu_threads=2,
                sampling='stochastic_private_rng', pair_id=CAMPAIGN + '-paired-u32', paired_slots=EVAL_COUNT,
                run_index_base=FIRST_RUN, report_directory=str(REPORT), python_output=str(OUTPUT),
                control_directory=str(CONTROL), material_root=str(MATERIALS), budget_path=str(LEDGER),
                active_h2_reserved_shifts=0, legacy_reserved_shifts=0, occupied_before=PREVIOUS_CAP,
                maximum_total_if_approved=PREVIOUS_CAP + TOTAL,
                mean_ranges={key: list(value) for key, value in MEAN_RANGES.items()},
                half_width_range=list(HALF_WIDTH_RANGE), heldout_orders=list(HOLDOUTS),
                evaluation_cases={name: dict(order=order, means=list(means), half_width=half_width)
                                  for name, (order, means, half_width) in EVAL_CASES.items()},
                protected_core=reference(CORE))


def _learner_configs():
    from business_graph_summary_v1.learner import default_config

    old_config = read(EXTENSION / 'prepared_001/suite.json')['learner_config']
    train_config = default_config(seed=seed(CAMPAIGN, 'initialization', 1), candidate_id=CANDIDATE,
                                  execution_profile=old_config['execution_profile'])
    require(dict(train_config, seed=old_config['seed']) == old_config, 'model, reward, or optimizer differs from historical settings')
    return train_config, old_config


def _phase_entry(cfg, ref):
    return dict(phase_id=cfg['phase_id'], phase_config=ref, policy_arm=cfg['policy_arm'],
                scenario_id=cfg['scenario_id'], update_index=cfg['arm_update'],
                role='train' if cfg['policy_arm'] == 'train' else 'eval',
                checkpoint_update=cfg['jobs'][0]['checkpoint_update'], python_output=cfg['python_output'])


def _write_rows(path, columns, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)


def _identity_rows(phases, refs):
    for cfg, ref in zip(phases, refs, strict=True):
        for job in cfg['jobs']:
            enriched = dict(job, config_sha256=ref['sha256'], policy_arm=cfg['policy_arm'],
                            assignment=_job_order(job))
            yield {key: enriched[key] for key in IDENTITY_COLUMNS}


def _job_order(job):
    expected = job['expected_rpc_config']
    ore = [(expected[f'{PREFIX}load.{site}.grade'], site) for site in range(4)
           if expected[f'{PREFIX}load.{site}.isOre'] == 1]
    labels = ['W'] * 4
    for (_, site), role in zip(sorted(ore, reverse=True), ('H', 'M', 'L'), strict=True):
        labels[site] = role
    return ''.join(labels)


def prepare():
    """Create the offline sealed package. AnyLogic Run owns authority and registration."""
    from mixed_java import verify_java, write_java

    require(not any(path.exists() for path in (PREPARED, OUTPUT, MATERIALS)),
            'new immutable preparation and experiment output paths required')
    train_config, old_config = _learner_configs()
    artifacts = _check_artifacts()
    settings = _suite_settings(train_config, old_config, artifacts)
    phases, jobs = _phase_documents(train_config['seed'], old_config['seed'])
    expanded = dict(settings, jobs=jobs)
    _check_inventory(expanded, phases)
    check_collisions(expanded)
    refs = []
    for cfg in phases:
        path = PREPARED / 'configs' / f"{cfg['phase_id']}.json"
        write(path, cfg)
        refs.append(reference(path))
    master = PREPARED / 'master_plan.csv'
    table = PREPARED / 'scenarios.csv'
    _write_rows(master, COLUMNS, jobs)
    helper = write_java(settings, jobs, master, table, PREPARED)
    checks = verify_java(helper, table)
    require(checks.get('compiled') is True and checks.get('rows_checked') == TOTAL
            and checks.get('simulation_started') is False, 'incomplete offline Java verification')
    suite = dict(settings, phases=[_phase_entry(cfg, ref) for cfg, ref in zip(phases, refs, strict=True)],
                 phase_configs=refs, source_pins=source_pins(), master_plan=reference(master),
                 scenario_table=reference(table), java_helper=reference(helper),
                 one_click_entry=reference(PLAN / 'mixed_bootstrap.py'), java_validation=checks,
                 phase_templates=[reference(TRAIN_TEMPLATE), reference(EVAL_TEMPLATE)])
    identities = PREPARED / 'identities.csv'
    _write_rows(identities, IDENTITY_COLUMNS, _identity_rows(phases, refs))
    suite['identities'] = reference(identities)
    write(SUITE, suite)
    write(PREPARED / 'launch_identity.json',
          dict(suite=reference(SUITE), python_output=str(OUTPUT), anylogic_event_root=str(MATERIALS),
               training_shifts=TRAINING_SHIFTS, evaluation_shifts=EVAL_SHIFTS, physical_shifts=TOTAL,
               first=next(_identity_rows(phases, refs)), identities=reference(identities),
               phases=[dict(phase, recording_id=cfg['recording_id'],
                            material_event_directory=cfg['jobs'][0]['material_event_directory'],
                            first_run=cfg['jobs'][0]['run_index'], last_run=cfg['jobs'][-1]['run_index'],
                            shifts=len(cfg['jobs'])) for phase, cfg in zip(suite['phases'], phases, strict=True)],
               old_checkpoint=artifacts['checkpoint'], java_helper=reference(helper),
               scenario_table=reference(table)))
    load_package()
    require(not any(path.exists() for path in (OUTPUT, MATERIALS, APPROVAL, LEDGER)),
            'offline preparation unexpectedly acquired a live experiment identity')
    return suite


def load_package():
    from mixed_java import scenario_table_bytes

    suite, phases = load_suite(SUITE)
    train, old = _learner_configs()
    settings = _suite_settings(train, old, _check_artifacts())
    require(all(suite.get(key) == value for key, value in settings.items()),
            'frozen mixed campaign scope, model, or output differs')
    require(suite['source_pins'] == source_pins(), 'sealed campaign implementation bytes changed')
    require(suite['phase_templates'] == [reference(TRAIN_TEMPLATE), reference(EVAL_TEMPLATE)],
            'train or evaluation phase template changed')
    require(suite['phases'] == [_phase_entry(cfg, ref) for cfg, ref in zip(phases, suite['phase_configs'], strict=True)],
            'frozen phase descriptors differ')
    for key, path in (
        ('master_plan', PREPARED / 'master_plan.csv'), ('scenario_table', PREPARED / 'scenarios.csv'),
        ('java_helper', PREPARED / 'Baseline1CampaignPlan.java'), ('identities', PREPARED / 'identities.csv'),
        ('one_click_entry', PLAN / 'mixed_bootstrap.py'),
    ):
        require(suite[key] == reference(path), 'frozen package artifact changed: ' + key)
    require(suite['phase_configs'] == [reference(PREPARED / 'configs' / f"{cfg['phase_id']}.json")
                                      for cfg in phases], 'noncanonical mixed phase file path')
    _check_inventory(suite, phases)
    with Path(suite['master_plan']['path']).open(newline='', encoding='utf-8') as stream:
        require(list(csv.DictReader(stream)) == [{key: str(job[key]) for key in COLUMNS} for job in suite['jobs']],
                'Java master and RPC job inventory differ')
    require(Path(suite['scenario_table']['path']).read_bytes() == scenario_table_bytes(suite['jobs']),
            'Java physical scenario table and RPC job inventory differ')
    with Path(suite['identities']['path']).open(newline='', encoding='utf-8') as stream:
        require(list(csv.DictReader(stream)) == [{key: str(value) for key, value in row.items()}
                                                 for row in _identity_rows(phases, suite['phase_configs'])],
                'human identity sheet and RPC job inventory differ')
    return suite, phases


if __name__ == '__main__':
    prepare()
    print(f'PREPARED_ONLY {SUITE}')
