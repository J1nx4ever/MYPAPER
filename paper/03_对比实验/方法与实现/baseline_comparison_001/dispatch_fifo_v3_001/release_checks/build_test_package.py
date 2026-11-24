"""Export this frozen 100-shift test and its checked project source files."""
import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


PLAN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLAN))
import v3_plan as plan


def build(output):
    suite = plan.load_suite()
    workspace = plan.ROOT.parents[1]
    proof_path = plan.REPORT / 'release_reproduction.json'
    proof = json.loads(proof_path.read_bytes())
    plan.require(proof['roster']['suite'] == plan.reference(plan.SUITE)
                 and proof['physical_shifts_started'] == 0, 'release proof does not bind this suite')
    paths = set(plan.PREPARED.rglob('*')) | set(PLAN.glob('*.py')) | set((PLAN/'release_checks').glob('*.py'))
    paths.update((PLAN/'README.md', proof_path, Path(plan.RETIRED_SUITE['path']),
                  plan.FORMAL/'verify_results.py', PLAN.parent/'all_recheck_001/verify_results.py'))
    for ref in suite['source_pins']:
        paths.add(Path(ref['path']))
    for name, digest in proof['environment']['runtime_sources'].items():
        path = workspace/name
        plan.require(plan.reference(path)['sha256'] == digest, f'release runtime source changed: {name}')
        paths.add(path)
    files = {str(p.relative_to(workspace)): p.read_bytes() for p in sorted(paths) if p.is_file()}
    index = dict(campaign_id=plan.CAMPAIGN, suite=plan.reference(plan.SUITE),
                 repository_root=str(workspace), offline_physical_shifts=0,
                 prerequisites=['existing project with historical budget and sealed V2 physical evidence',
                                'baseline1 AnyLogic model, anylogic-env and JDK at project paths'],
                 guide=str((PLAN/'README.md').relative_to(workspace)),
                 files={name:sha256(data).hexdigest() for name,data in files.items()})
    files['PACKAGE.json'] = plan.encoded(index)
    stream = io.BytesIO()
    with ZipFile(stream, 'w') as archive:
        for name, content in sorted(files.items()):
            entry = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, content)
    with ZipFile(stream) as archive:
        plan.require(set(archive.namelist()) == set(files) and
                     all(archive.read(name) == content for name, content in files.items()), 'package readback failed')
    data = stream.getvalue()
    if output.exists():
        plan.require(output.read_bytes() == data, 'different package already exists at destination')
    else:
        with output.open('xb') as target:
            target.write(data)
    return dict(package=plan.reference(output), files=len(files), bytes=len(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=plan.REPORT / 'dispatch_fifo_paired100_002.zip')
    print(json.dumps(build(parser.parse_args().output)))
