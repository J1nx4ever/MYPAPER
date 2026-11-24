"""Build a portable, byte-checked decision replay and prove it outside the repository."""
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import zipfile

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
PLAN = HERE.parent
ROOT = PLAN.parents[4]
REPORTS = ROOT / 'reports' / PLAN.parent.relative_to(ROOT / 'change_plans')
DEST = REPORTS / 'dispatch_fifo_v3_002' / 'dispatch_fifo_replay.zip'
GOLDEN = REPORTS / 'dispatch_fifo_v3_001' / 'offline_review.json'
GOLDEN_SHA = '050a99af075e8967d7263db73586c20c95e94fe7eb8b6ba21fb8f268b93f3817'
SUITE = PLAN / 'prepared_002' / 'suite.json'
ARCHIVES = ROOT / 'runs/bg128-continuous-mixed-20261120-s02/evaluation/nominal/new/material'
SCHEMA = ROOT / 'python/observation_schema.json'
SCHEMA_SHA = 'c242ce76856d1cd929af738da3942bab312749712603626d6c2747ac47d12637'
EXPECTED_FIELDS = ('run_index', 'frame', 'archive_sha256', 'observation_sha256', 'action', 'lp_status',
                   'fifo_root_feasible', 'fifo_selected_feasible', 'fifo_selected_actions',
                   'fifo_selected_truck_ids', 'fifo_roots_completed', 'fifo_root_cost',
                   'fifo_selected_cost', 'fifo_selected_industrial_loss')
PACKAGES = ('numpy', 'scipy', 'torch')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def checked_bytes(path, digest):
    data = path.read_bytes()
    check(sha(data) == digest, f'changed sealed input {path.name}')
    return data


def imported_sources():
    # Use a separate interpreter to observe only the policy import graph. Standard
    # library and installed third-party packages are supplied by the environment.
    probe = """
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [sys.argv[2], str(root)]
import fifo_policy
paths = set()
for module in list(sys.modules.values()):
    origin = getattr(getattr(module, '__spec__', None), 'origin', None)
    if isinstance(origin, str) and origin.endswith('.py'):
        path = Path(origin)
        if path.is_absolute() and path.is_relative_to(root) and path.is_file():
            paths.add(path.relative_to(root).as_posix())
print(json.dumps(sorted(paths)))
"""
    result = subprocess.run([sys.executable, '-I', '-B', '-c', probe, str(ROOT), str(PLAN)], check=True,
                            text=True, capture_output=True)
    paths = [ROOT / name for name in json.loads(result.stdout)]
    check(set(PLAN.glob('*.py')).intersection(paths) == {PLAN / name for name in
          ('fifo_policy.py', 'v3_flow.py', 'v3_matching.py')}, 'unexpected local controller dependency graph')
    return paths


def freeze_frames():
    golden = json.loads(checked_bytes(GOLDEN, GOLDEN_SHA))
    rows = [{key: row[key] for key in EXPECTED_FIELDS} for row in golden['rows']]
    check(len(rows) == 24 and len({row['observation_sha256'] for row in rows}) == 22,
          'unexpected sealed reference frame inventory')
    states = {}
    for run in sorted({row['run_index'] for row in rows}):
        file = ARCHIVES / f'run_{run}.h5'
        seal = json.loads(file.with_suffix('.h5.seal.json').read_bytes())
        check(seal['status'] == 'complete', 'historical material archive missing full completion')
        checked_bytes(file.with_suffix('.h5.manifest.json'), seal['manifest_sha256'])
        sealed = checked_bytes(file, seal['hdf5_sha256'])
        with h5py.File(io.BytesIO(sealed), 'r') as hdf:
            for row in (row for row in rows if row['run_index'] == run):
                state = hdf['states'][row['frame']]
                check(state.dtype == np.float32 and state.shape == (1519,) and
                      sha(state.tobytes()) == row['observation_sha256'] and row['archive_sha256'] == seal['hdf5_sha256'],
                      f'frame differs from frozen archive {run}/{row["frame"]}')
                states[run, row['frame']] = state
    data = io.BytesIO()
    np.save(data, np.stack([states[row['run_index'], row['frame']] for row in rows]), allow_pickle=False)
    return rows, data.getvalue()


def build():
    suite_bytes = SUITE.read_bytes()
    identity = json.loads(SUITE.with_name('launch_identity.json').read_bytes())
    check(sha(suite_bytes) == identity['suite']['sha256'], 'release suite identity differs')
    suite = json.loads(suite_bytes)
    pins = {ref['path']: ref['sha256'] for ref in suite['source_pins']}
    rows, arrays = freeze_frames()
    sources = imported_sources()
    files = {}
    for path in sources:
        data = path.read_bytes()
        if str(path) in pins or path.parent == PLAN:
            check(sha(data) == pins[str(path)], f'changed production decision code {path.name}')
        files['src/' + path.relative_to(ROOT).as_posix()] = data
    files['src/python/observation_schema.json'] = checked_bytes(SCHEMA, SCHEMA_SHA)
    files['expected.json'] = encoded(rows)
    files['observations.npy'] = arrays
    files['replay.py'] = (HERE / 'replay_bundle.py').read_bytes()
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    files['requirements.txt'] = ''.join(f'{name}=={version}\n' for name, version in versions.items()).encode()
    files['README.md'] = f"""# DISPATCH-inspired FIFO 默认策略的离线复现

本包包含实际派车算法及其导入的项目源码、1519 维观测 schema 和 24 个历史决策帧。这些帧来自三个已封存班次的每个品位窗首尾，共 22 种不同的观测。每帧新建策略，以生产参数 800 ms、900 秒重规划、三车匹配、最多 12 次组合试算运行。在本批输入上逐帧对照动作、候选车辆和装点前缀、合规性及成本。该离线检查不产生物理班次。

在装有 Python {platform.python_version()}、NumPy {versions['numpy']}、SciPy {versions['scipy']}、PyTorch {versions['torch']} 的环境中解压，进入解压目录后运行：

```bash
python -I -B replay.py
```

脚本先校验文件字节，随后执行全部 24 帧并输出比对结果和本机规划耗时。满载或较慢机器可能触发 800 ms 保护，此时脚本会列出首个涉及的运行号、帧号和规划结果。预期结果保存在 `expected.json`，环境版本在 `requirements.txt`，源码和输入 SHA256 在 `manifest.json`。
""".encode('utf-8')
    metadata = dict(campaign_id=suite['campaign_id'], physical_suite_sha256=sha(suite_bytes),
                    policy_directory=PLAN.relative_to(ROOT).as_posix(),
                    policy_options={k: suite['method'][k] for k in ('planning_ms', 'replan_seconds', 'match_trucks',
                                                                    'cohort_trial_limit')},
                    golden_review_sha256=GOLDEN_SHA, python=platform.python_version(), packages=versions,
                    files={name: sha(data) for name, data in files.items()})
    files['manifest.json'] = encoded(metadata)
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        archive = temp / DEST.name
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as output:
            for name, data in sorted(files.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                output.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
        isolated = temp / 'independent'
        with zipfile.ZipFile(archive) as reader:
            reader.extractall(isolated)
        env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
        process = subprocess.run([sys.executable, '-I', '-B', str(isolated / 'replay.py')], cwd=temp,
                                 env=env, text=True, capture_output=True)
        check(process.returncode == 0, 'isolated replay failed\n' + process.stderr)
        result = json.loads(process.stdout)
        check(result['status'] == 'PASS' and result['decisions_matched'] == 24, 'isolated replay incomplete')
        payload = archive.read_bytes()
    DEST.parent.mkdir(exist_ok=True)
    if DEST.exists():
        check(DEST.read_bytes() == payload, 'a different bundle already exists at the release destination')
    else:
        with tempfile.NamedTemporaryFile(dir=DEST.parent, delete=False) as output:
            staging = Path(output.name)
            output.write(payload)
        try:
            staging.chmod(0o644)
            os.link(staging, DEST)
        finally:
            staging.unlink()
    return dict(path=str(DEST), bundle_sha256=sha(payload), project_source_files=len(sources),
                total_bytes=len(payload), isolated_replay=result)


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
