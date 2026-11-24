"""Reproduce the frozen roster, Java and offline DISPATCH decisions without launching AnyLogic."""
import argparse
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


PLAN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLAN))

import v3_plan as plan
from prepare import IDENTITY_COLUMNS
from review_offline import review
from v3_java import master_bytes, render, scenario_table_bytes, verify_java


GOLDEN_SHA = '050a99af075e8967d7263db73586c20c95e94fe7eb8b6ba21fb8f268b93f3817'
PACKAGES = ('numpy', 'scipy', 'h5py', 'grpcio', 'protobuf', 'torch', 'pytest')


def regression_tests():
    # Use the active test files only. No archive or offline reproducer is a pytest input.
    directories = (plan.ROOT / 'traditional_dispatch_v1', plan.ROOT / 'traditional_dispatch_v2',
                   PLAN.parent / 'dispatch_inspired_001', plan.FORMAL, PLAN, PLAN / 'release_checks')
    tests = [str(p.relative_to(plan.ROOT)) for directory in directories for p in sorted(directory.glob('test_*.py'))]
    command = [plan.PYTHON, '-B', '-m', 'pytest', '-p', 'no:cacheprovider', '-q', *tests]
    result = subprocess.run(command, cwd=plan.ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError('offline regression suite failed\n' + result.stdout)
    summary = result.stdout.strip().splitlines()[-1]
    passed = re.search(r'\b(\d+) passed\b', summary)
    plan.require(passed and int(passed[1]) > 0 and 'skipped' not in summary and 'xfailed' not in summary,
                 'regression suite did not pass all tests: ' + summary)
    return dict(command=command, cwd=str(plan.ROOT), passed=int(passed[1]), summary=summary)


def check_roster(suite):
    prepared = plan.PREPARED
    card_path = prepared / 'launch_identity.json'
    card = json.loads(card_path.read_bytes())
    plan.require(card['suite'] == plan.reference(plan.SUITE), 'identity card suite fingerprint differs')
    for key in ('campaign_id', 'java_helper', 'run_seed', 'method', 'previous_formal_suite', 'protected_core',
                'evaluation_slots', 'physical_shifts', 'training_shifts', 'optimizer_updates', *plan.LAYOUT):
        plan.require(card[key] == suite[key], f'identity card differs on {key}')
    with Path(suite['run_seed']['path']).open(encoding='ascii', newline='') as stream:
        reader = csv.DictReader(stream)
        plan.require(tuple(reader.fieldnames or ()) == IDENTITY_COLUMNS, 'run_seed.csv headers differ')
        csv_rows = list(reader)
    _, anchors = plan.source_contract()
    jobs = suite['jobs']
    plan.require(len(jobs) == len(anchors) == len(card['jobs']) == len(csv_rows) == 100,
                 'exactly 100 physical jobs and reference slots required')
    for index, (job, old, csv_row, card_row) in enumerate(zip(jobs, anchors, csv_rows, card['jobs'], strict=True)):
        plan.require(job['ordinal'] == job['pair_ordinal'] == old['pair_ordinal'] == index and
                     job['experiment_seed'] == old['experiment_seed'] and job['actor_sampling_seed'] == 0 and
                     job['actor_sampling_seed_consumed'] is False, f'wrong deterministic pairing at slot {index}')
        row = {k: job[k] for k in IDENTITY_COLUMNS if k in job}
        row.update(config_path=job['phase_config']['path'], config_sha256=plan.reference(job['phase_config']['path'])['sha256'],
                   anylogic_event_directory=job['event_directory'], v2_reference_run_index=old['run_index'],
                   v2_recording_id=old['recording_id'], v2_config_sha256=plan.reference(old['phase_config']['path'])['sha256'])
        plan.require(set(row) == set(IDENTITY_COLUMNS) and
                     json.dumps(row, sort_keys=True) == json.dumps(card_row, sort_keys=True) and
                     {k: str(v) for k, v in row.items()} == csv_row, f'identity mismatch at slot {index}')
    master = Path(suite['master_plan']['path'])
    scenarios = Path(suite['scenario_table']['path'])
    helper = Path(suite['java_helper']['path'])
    plan.require(master_bytes(jobs) == master.read_bytes() and scenario_table_bytes(jobs) == scenarios.read_bytes(),
                 'regenerated Java sidecar differs')
    plan.require(render(jobs, master, scenarios).encode('utf-8') == helper.read_bytes(),
                 'regenerated Java helper bytes differ')
    java = verify_java(helper, scenarios, jobs)
    plan.require(java == card['java_verification'] and java['rows_checked'] == java['paired_environment_rows_checked'] == 100
                 and java['simulation_started'] is False, 'Java compile or 100-slot proof differs')
    return dict(paired_rows=100, distinct_environment_seeds=len({r['experiment_seed'] for r in jobs}),
                phase_hashes_checked=100, identity=plan.reference(card_path), suite=plan.reference(plan.SUITE),
                run_seed=suite['run_seed'], protected_core=suite['protected_core'], java_helper=suite['java_helper'],
                java_verification=java)


def check_offline():
    path = plan.REPORT.parent / 'dispatch_fifo_v3_001' / 'offline_review.json'
    golden = plan.load_reference(dict(path=str(path), sha256=GOLDEN_SHA))
    current = review()
    identity = ('run_index', 'frame', 'archive_sha256', 'observation_sha256')
    decisions = ('action', 'kind', 'lp_status', 'fifo_selected_actions', 'fifo_selected_truck_ids',
                 'fifo_root_feasible', 'fifo_selected_feasible')
    costs = ('fifo_root_cost', 'fifo_selected_cost')
    plan.require(current['frames'] == golden['frames'] == 24 and
                 current['distinct_observations'] == golden['distinct_observations'] == 22,
                 'unexpected historical decision snapshot inventory')
    # A cold planner has its own measured wall time on each machine. A changed
    # wall-guard outcome still fails reproduction even when the action ties.
    plan.require(current['planning_timeouts'] == 0, 'a historical snapshot triggered the wall guard')
    for now, saved in zip(current['rows'], golden['rows'], strict=True):
        where = f"{now['run_index']}/{now['frame']}"
        for key in (*identity, *decisions):
            value = list(now[key]) if isinstance(now[key], tuple) else now[key]
            plan.require(value == saved[key], f'historical snapshot {where} differs on {key}: {value} vs {saved[key]}')
        for key in costs:
            plan.require(math.isclose(now[key], saved[key], rel_tol=1e-10, abs_tol=1e-10),
                         f'historical snapshot {where} differs on {key}: {now[key]} vs {saved[key]}')
        plan.require(now['fifo_candidates'] == now['fifo_roots_completed'] and now['kind'] == 'planned'
                     and now['fifo_guidance_error'] is None and not now['fifo_guard_restored_root'],
                     f'historical snapshot {where} lost a legal root, LP guidance or the completed candidate search')
    return dict(golden=plan.reference(path), checked_fields=[*identity, *decisions, *costs], **current)


def environment():
    # Include shared RPC, protobuf and ledger code imported by the real runner.
    import v3_runtime
    import launch

    repo = plan.CORE.parents[1]
    paths = set()
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file and file.endswith('.py'):
            path = Path(file).resolve()
            if path.is_relative_to(repo) and path.is_file():
                paths.add(path)
    packages = {key: importlib.metadata.version(key) for key in PACKAGES}
    versions = {}
    for program in ('java', 'javac'):
        result = subprocess.run([program, '-version'], capture_output=True, text=True, check=True)
        versions[program] = (result.stdout + result.stderr).strip()
    return dict(python_version=platform.python_version(), executable=sys.executable, packages=packages,
                java_versions=versions, platform=platform.platform(), machine=platform.machine(),
                runtime_sources={str(p.relative_to(repo)): plan.reference(p)['sha256'] for p in sorted(paths)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='write the full JSON to a new path, or print it on stdout')
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('output file already exists; choose a new path')
    plan.require(Path(sys.executable).resolve() == Path(plan.PYTHON).resolve(), 'use the project anylogic-env interpreter')
    suite = plan.load_suite()
    report = dict(campaign_id=plan.CAMPAIGN, scope='offline_only', reproduction_code=plan.reference(__file__),
                  roster=check_roster(suite), regression_tests=regression_tests(), historical_replay=check_offline(),
                  environment=environment(), physical_shifts_started=0)
    data = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as output:
            output.write(data)
        print(json.dumps(dict(status='PASS', report=plan.reference(args.output))))
    else:
        print(data, end='')


if __name__ == '__main__':
    main()
