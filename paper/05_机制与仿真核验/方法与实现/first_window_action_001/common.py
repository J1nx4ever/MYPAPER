"""Scope and artifact helpers for the four-shift material experiment."""
import hashlib
import json
from pathlib import Path
import sys

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[3]
sys.path.insert(0, str(ROOT))

from business_graph_summary_campaign_v1.plan_io import encoded, reference, sha
from business_graph_summary_v1.contract import require
from online_v1_1.operations import _publish

CID = 'bg128-first-window-action-20261119-001'
PHASE = 's01_BG128_first_window'
RECORDING = CID + '-' + PHASE
PREPARED = PLAN / 'prepared_002'
RUN = ROOT / 'runs' / CID
MATERIALS = ROOT / 'runs' / (CID + '_materials') / PHASE
REPORT = ROOT / 'reports/token_attention_001/business_graph_summary_009/first_window_action_001'
OLD_PLAN = PLAN.parent / 'checkpoint_evaluation_001'
OLD_REPORT = REPORT.parent / 'checkpoint_evaluation_001'
LOW_PLAN = PLAN.parent / 'optimization_stability_001/paired_evaluation_001'
LOW_REPORT = REPORT.parent / 'optimization_stability_001/paired_evaluation_001'
CORE = ROOT.parents[1] / 'python/episode_budget_ppo.py'
CORE_SHA = '8e4cfa31c622f5e33756e304eea3f7526547db72ae93f7d9f00808845d61ef96'
CASES = (9950012, 9950029)
RUN_IDS = (9980001, 9980002, 9980003, 9980004)
BEHAVIOR_KIND = 'frozen_actor_diagnostic_single_action'
CHECKPOINT_SHA = 'a9acc888413ce0861240c04c396a239d9c26e48290eddc9f2c14dd9c97e6afd5'


def read(path):
    return json.loads(Path(path).read_bytes())


def checked(ref):
    path = Path(ref['path'])
    require(reference(path) == ref, 'artifact changed: ' + str(path))
    return path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _publish(path, encoded(value))
    return reference(path)


def state_sha(values):
    return hashlib.sha256(values.tobytes(order='C')).hexdigest()


def source_records():
    sources = []
    for suite_path, audit_path in (
        (OLD_PLAN/'prepared_001/suite.json', OLD_REPORT/'completion_independent_audit.json'),
        (LOW_PLAN/'prepared_001/suite.json', LOW_REPORT/'completion_independent_audit.json'),
    ):
        suite, audit = read(suite_path), read(audit_path)
        require(audit['suite'] == reference(suite_path) and audit['status'].startswith('PASS'),
                'source independent evaluation audit differs')
        sources.append((suite, {row['run_index']:row for row in audit['per_run']}))
    return sources
