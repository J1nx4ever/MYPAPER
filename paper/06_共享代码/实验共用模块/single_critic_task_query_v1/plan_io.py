"""Compact immutable disk configs, expanded only inside the checked reader."""
import copy
import hashlib
import json
from pathlib import Path
from .contract import require

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'change_plans/token_attention_001/single_critic_task_query_008/engineering_001'
REPORT = ROOT / 'reports/token_attention_001/single_critic_task_query_008/engineering_001'

def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))

def load_reference(ref):
    require(reference(ref['path']) == ref, 'frozen artifact byte SHA changed: ' + ref['path'])
    return json.loads(Path(ref['path']).read_bytes())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(encoded(value))

def load_phase(path):
    value = json.loads(Path(path).read_bytes())
    if 'common_reference' not in value:  # synthetic sealed fixture support
        return value
    common = load_reference(value.pop('common_reference'))
    require(not set(common).intersection(value), 'phase cannot override common contract')
    common.update(value)
    defaults = common.pop('job_defaults')
    result = []
    for job in common['jobs']:
        require(not set(defaults).intersection(job), 'job cannot override common RPC contract')
        result.append(dict(defaults, **job))
    common['jobs'] = result
    return common

def load_suite(path):
    value = json.loads(Path(path).read_bytes())
    require('jobs' not in value, 'compact suite cannot hide a second job inventory')
    phases = []
    for ref in value['phase_configs']:
        require(reference(ref['path']) == ref, 'phase config byte SHA changed')
        phases.append(load_phase(ref['path']))
    value['jobs'] = [job for phase in phases for job in phase['jobs']]
    return value, phases
