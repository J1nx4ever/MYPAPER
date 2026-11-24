"""Reproduce the fixed model selection for the 100-slot nominal comparison."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import fmean
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from business_graph_summary_campaign_v1.plan_io import encoded, load_reference, reference, write
from single_critic_task_query_v1.targets import (
    ORE_TARGET_TONS, ORE_WEIGHT, WAIT_REFERENCE_SECONDS, WAIT_WEIGHT, WASTE_TARGET_TONS,
    WASTE_WEIGHT, grade_cost,
)

SOURCES = {
    's01': ('bg128-continuous-mixed-20261119-001',
            'e32326901f5c070c5bd64160c8978ea0b491947da7eb159f380c8ddef945bc9d'),
    's02': ('bg128-continuous-mixed-20261120-s02',
            '45d7735b6612b55404bdf240f2c7885667f5e80ba07ffaea07aa38eefdb808fc'),
    's03': ('bg128-continuous-mixed-20261120-s03',
            '799afbff30c8ec2ba52d8b54747b9bed0fae78dac21ea4ad8bd8063b962d81ee'),
}
HOLDOUT = ('held_order_1', 'held_order_2', 'new_numeric', 'joint_shift', 'waste_swap')
SCENARIOS = ('nominal', *HOLDOUT, 'grade_swap')
CHECKPOINT_SHA = '67b7b2c7104079836ccef8f5747e7aebb4e9e5a92dcaac67c94e42412f104f41'
ACTOR_SHA = 'e1575fec566c647a882c63934ec57c908cef28d43f076431ef9103cc9b73e0a9'
ACTOR_DIGEST = '189a2c571c4bcf0fbe978330d92dddb84f3b196b753f7ed93e533988d4c59abd'
UPDATE = 32
SHIFTS_PER_SCENARIO = 100
OUT = (ROOT / 'reports/token_attention_001/business_graph_summary_009/baseline_comparison_001'
       / 'paired_100_001/checkpoint_selection.json')
MEANS = ('combined_cost', 'grade_cost', 'production_cost', 'waiting_cost',
         'ore_tons', 'waste_tons', 'total_wait_seconds')
FLAGS = ('complete_task', 'all_four_windows_compliant', 'ore_target_met', 'waste_target_met')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(ref, path):
    require(reference(path) == ref, f'CSV bytes or path differ from the evaluation seal: {path}')
    with io.StringIO(path.read_text(), newline='') as stream:
        return list(csv.DictReader(stream))


def metrics(shifts, windows):
    valid = [float(row['grade']) for row in windows if row['empty'] == 'False']
    counts = {key: sum(row[key] == 'True' for row in shifts) for key in FLAGS}
    return {
        'shifts': len(shifts),
        'means': {key: fmean(float(row[key]) for row in shifts) for key in MEANS},
        'counts': counts,
        'rates': {key: count / len(shifts) for key, count in counts.items()},
        'valid_grade_windows': len(valid),
        'empty_windows': len(windows) - len(valid),
        'grade_rmse': math.sqrt(fmean((grade - .27) ** 2 for grade in valid)) if valid else None,
    }


def verify_physical_rows(shifts, windows):
    by_run = {}
    for window in windows:
        by_run.setdefault(window['run_index'], []).append(window)
    require(len(windows) == 4 * len(shifts) and set(by_run) == {r['run_index'] for r in shifts},
            'evaluation window/shift inventory differs')
    for row in shifts:
        for key in MEANS:
            require(math.isfinite(float(row[key])), f'nonfinite physical value: {key}')
        actual = sorted(by_run[row['run_index']], key=lambda w: w['window'])
        require([w['window'] for w in actual] == list('0123'), 'missing or duplicate grade window')
        require(all(w[key] == row[key] for w in actual
                    for key in ('environment_seed', 'actor_sampling_seed')),
                'window/shift seeds differ')
        grades = []
        for window in actual:
            tons = float(window['tons'])
            empty = window['empty'] == 'True'
            grade = None if empty else float(window['grade'])
            if not empty:
                require(math.isclose(grade, float(window['metal_tons']) / tons,
                                     rel_tol=0.0, abs_tol=1e-12),
                        'grade does not match settled mass')
            compliant = not empty and .26 - 1e-12 <= grade <= .28 + 1e-12
            require(window['compliant'] == str(compliant), 'window compliance differs')
            grades.append(grade_cost(grade, empty=empty))
        ore, waste, wait = (float(row[k]) for k in ('ore_tons', 'waste_tons', 'total_wait_seconds'))
        grade, production, waiting = (math.fsum(grades),
                                      ORE_WEIGHT * max(0.0, 1 - ore / ORE_TARGET_TONS)
                                      + WASTE_WEIGHT * max(0.0, 1 - waste / WASTE_TARGET_TONS),
                                      WAIT_WEIGHT * wait / WAIT_REFERENCE_SECONDS)
        for key, expected in (('grade_cost', grade), ('production_cost', production),
                              ('waiting_cost', waiting), ('combined_cost', grade + production + waiting)):
            require(math.isclose(expected, float(row[key]), rel_tol=1e-10, abs_tol=1e-10),
                    f'physical cost does not match the frozen objective: {key}')
        compliant = all(w['compliant'] == 'True' for w in actual)
        ore_met, waste_met = ore >= ORE_TARGET_TONS, waste >= WASTE_TARGET_TONS
        for key, expected in zip(FLAGS, (compliant and ore_met and waste_met,
                                        compliant, ore_met, waste_met), strict=True):
            require(row[key] == str(expected), f'physical threshold flag differs: {key}')


def inspect_phase(run, scenario, phase, checkpoint, manifest):
    source = run / 'evaluation' / scenario / 'new'
    require(phase['checkpoint'] == checkpoint
            and Path(phase['report']['path']) == source / 'evaluation_complete.json',
            'scenario/evaluation checkpoint reference differs')
    receipt = load_reference(phase['report'])
    require(receipt['source_kind'] == 'fixed_actor' and receipt['role'] == 'independent_evaluation'
            and receipt['scenario_id'] == scenario and receipt['policy_arm'] == 'new'
            and receipt['training_state_unchanged'] and receipt['n_shifts'] == SHIFTS_PER_SCENARIO
            and receipt['checkpoint_provenance'] == checkpoint
            and receipt['fingerprints']['actor'] == manifest['actor_digest']
            and all(receipt[key] == 0 for key in ('evaluation_optimizer_updates', 'optimizer_updates',
                                                 'gradient_updates', 'actor_updates', 'critic_updates')),
            'evaluation receipt violates the frozen fixed-Actor scope')
    actor = source / 'actor.pt'
    actor_ref = receipt['actor_artifact']
    actor_manifest_ref = reference(actor.with_name('actor.pt.manifest.json'))
    actor_manifest = load_reference(actor_manifest_ref)
    require(reference(actor) == actor_ref and actor_manifest['sha256'] == actor_ref['sha256']
            and actor_manifest['bytes'] == actor.stat().st_size
            and all(actor_manifest[key] == manifest[key] for key in
                    ('actor_digest', 'update_index', 'candidate_id', 'config_sha256', 'execution_profile')),
            'Actor archive differs from the training checkpoint identity')
    reports = phase['reports']
    summary = load_reference(reports['summary'])
    require(summary['n_shifts'] == SHIFTS_PER_SCENARIO
            and summary['optimizer_updates'] == summary['gradient_updates'] == 0
            and summary['scenario_id'] == scenario and summary['checkpoint_provenance'] == checkpoint
            and summary['action_selection'] == 'stochastic_private_per_shift_generator',
            'fixed independent evaluation summary differs')
    shifts = read_rows(reports['per_shift'], source / 'reports/per_shift.csv')
    windows = read_rows(reports['per_window'], source / 'reports/per_window.csv')
    config = load_reference(receipt['phase_config'])
    require(config['evaluation_only'] and config['scenario_id'] == scenario and config['optimizer_steps'] == 0
            and len(shifts) == len(config['jobs']) == SHIFTS_PER_SCENARIO, 'evaluation slot scope differs')
    by_run = {int(row['run_index']): row for row in shifts}
    require(len(by_run) == SHIFTS_PER_SCENARIO, 'duplicate evaluation run')
    slot_fields = ('pair_ordinal', 'scene_id', 'runtime_scene_sha256', 'expected_rpc_config',
                   'experiment_seed', 'actor_sampling_seed')
    slots = []
    for ordinal, job in enumerate(sorted(config['jobs'], key=lambda item: item['pair_ordinal'])):
        row = by_run[job['run_index']]
        require(job['pair_ordinal'] == ordinal and job['scene_id'] == scenario
                and job['checkpoint_update'] == UPDATE
                and (int(row['environment_seed']), int(row['actor_sampling_seed']))
                == (job['experiment_seed'], job['actor_sampling_seed']), 'fixed evaluation slot identity differs')
        slots.append({key: job[key] for key in slot_fields})
    require(len({(s['experiment_seed'], s['actor_sampling_seed']) for s in slots})
            == SHIFTS_PER_SCENARIO, 'duplicate paired seed slot')
    verify_physical_rows(shifts, windows)
    output = metrics(shifts, windows)
    require(math.isclose(output['means']['combined_cost'], receipt['metrics']['combined_cost']['mean'],
                         abs_tol=1e-12), 'recomputed combined cost differs from sealed result')
    output.update(
        receipt=phase['report'], reports=reports, phase_config=receipt['phase_config'],
        actor=actor_ref, actor_manifest=actor_manifest_ref,
        pairing_sha256=hashlib.sha256(encoded(slots)).hexdigest(),
    )
    return output, shifts, windows


def inspect_seed(seed, source):
    directory, seal_sha = source
    run = ROOT / 'runs' / directory
    seal_ref = {'path': str(run / 'suite_complete.json'), 'sha256': seal_sha}
    seal = load_reference(seal_ref)
    checkpoint = seal['new_checkpoint']
    path = run / 'arms/s01_BG128/updates/0032/checkpoint.pt'
    require(seal['status'] == 'COMPLETE' and seal['training_shifts'] == 1024
            and seal['training_updates'] == UPDATE and seal['evaluation_optimizer_updates'] == 0
            and seal['evaluation_shifts'] == (1400 if seed == 's01' else 700)
            and len(seal['training_checkpoints']) == UPDATE
            and seal['training_checkpoints'][-1] == checkpoint and reference(path) == checkpoint,
            'training completion scope or selected checkpoint bytes differ')
    manifest_ref = reference(path.with_name('checkpoint.pt.manifest.json'))
    manifest = load_reference(manifest_ref)
    require(manifest['update_index'] == UPDATE and manifest['bytes'] == path.stat().st_size
            and manifest['sha256'] == checkpoint['sha256'], 'training checkpoint manifest differs')
    for update, item in enumerate(seal['training_checkpoints'], 1):
        expected = path.parents[1] / f'{update:04}' / 'checkpoint.pt'
        require(Path(item['path']) == expected and expected.is_file(), 'training checkpoint inventory differs')

    phases = {Path(phase['report']['path']).parent: phase for phase in seal['phase_results']
              if phase['checkpoint'] == checkpoint}
    require(len(phases) == len(SCENARIOS), 'mixed final evaluation scenario inventory differs')
    outcomes = {}
    holdout_shifts, holdout_windows = [], []
    for scenario in SCENARIOS:
        phase = phases[run / 'evaluation' / scenario / 'new']
        outcome, shifts, windows = inspect_phase(run, scenario, phase, checkpoint, manifest)
        outcomes[scenario] = outcome
        if scenario in HOLDOUT:
            holdout_shifts.extend(shifts)
            holdout_windows.extend(windows)
    require(len({value['actor']['sha256'] for value in outcomes.values()}) == 1,
            'scenario evaluations used different Actor archive bytes')
    return {
        'seed': seed, 'campaign': directory, 'suite_completion': seal_ref, 'checkpoint': checkpoint,
        'checkpoint_manifest': manifest_ref, 'actor_digest': manifest['actor_digest'],
        'checkpoint_config_sha256': manifest['config_sha256'],
        'training_shifts': 1024, 'saved_updates': list(range(1, UPDATE + 1)),
        'independently_evaluated_updates': [UPDATE], 'independent_evaluation_optimizer_updates': 0,
        'scenario_evaluations': outcomes, 'nominal': outcomes['nominal'],
        'common_holdout': metrics(holdout_shifts, holdout_windows),
    }


def build_selection():
    candidates = [inspect_seed(seed, source) for seed, source in SOURCES.items()]
    for scenario in SCENARIOS:
        require(len({entry['scenario_evaluations'][scenario]['pairing_sha256']
                     for entry in candidates}) == 1, f'cross-seed RPC/seed slots differ: {scenario}')
    candidates.sort(key=lambda entry: (entry['nominal']['means']['combined_cost'], entry['seed']))
    selected = candidates[0]
    actor = selected['nominal']['actor']
    require(selected['seed'] == 's02' and selected['checkpoint']['sha256'] == CHECKPOINT_SHA
            and actor['sha256'] == ACTOR_SHA and selected['actor_digest'] == ACTOR_DIGEST,
            'selected Actor differs from the frozen pre-evaluation choice')
    return {
        'schema': 'bg128-nominal-paired-100-checkpoint-selection-v1',
        'selection_rule': {
            'eligible': 'mixed BG128 s01, s02, and s03 with sealed 7 x 100 fixed stochastic u32 evaluations',
            'selection_data': 'previously observed common nominal evaluation slots',
            'primary': 'minimize mean frozen whole-shift joint cost on the 100 historical nominal slots',
            'tie_break': 'training seed label in ascending order',
            'common_holdout_scenarios': list(HOLDOUT),
            'holdouts_and_grade_swap': 'supporting evidence only; not used to rank checkpoints',
        },
        'scientific_scope': (
            'The selected policy is conditional on one trained checkpoint. '
            'The new paired evaluation uses a fresh seed block and does not select models.'
        ),
        'pairing': {
            'slots_per_scenario': SHIFTS_PER_SCENARIO,
            'matched_fields': ['pair_ordinal', 'scene_id', 'runtime_scene_sha256',
                               'expected_rpc_config', 'experiment_seed', 'actor_sampling_seed'],
            'scenarios': list(SCENARIOS),
        },
        'selected_seed': selected['seed'], 'selected_update': UPDATE,
        'selected_checkpoint': selected['checkpoint'],
        'selected_checkpoint_manifest': selected['checkpoint_manifest'],
        'selected_checkpoint_config_sha256': selected['checkpoint_config_sha256'],
        'selected_actor': actor, 'selected_actor_manifest': selected['nominal']['actor_manifest'],
        'selected_actor_digest': selected['actor_digest'], 'candidates': candidates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='verify the existing byte-identical selection report')
    args = parser.parse_args()
    evidence = build_selection()
    if args.check or OUT.exists():
        require(OUT.read_bytes() == encoded(evidence), f'selection report differs: {OUT}')
    else:
        write(OUT, evidence)
    print(f"{evidence['selected_seed']}/u{evidence['selected_update']} {OUT}")


if __name__ == '__main__':
    main()
