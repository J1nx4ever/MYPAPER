"""Render a Baseline 1 Java identity gate from the unchanged audited template.

Rendering only reads a frozen CSV and returns Java text. It never starts Java,
AnyLogic, an RPC listener or a physical job, and never edits an ALP file.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re

COLUMNS = ('ordinal', 'actor_seed', 'group', 'role', 'source_recording_id', 'recording_id',
           'run_index', 'experiment_seed', 'actor_sampling_seed', 'event_directory',
           'event_bundle', 'python_output', 'target_training_update', 'checkpoint_update')
NATIVE_GROUPS = ('P0', 'P1', 'P2', 'P4')
GROUPS = frozenset((*NATIVE_GROUPS, 'BASELINE'))
ROLES = frozenset(('calibration', 't0', 'train', 'monitor', 'final'))
TEMPLATE = Path(__file__).resolve().parents[1] / 'change_plans' / 'stage8_seed1_one_click_001' / 'H2CampaignPlan.java.template'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _absolute(value, name):
    value = str(value)
    path = Path(value)
    _require(path.is_absolute() and str(path) == value and
             all(part not in ('.', '..') for part in path.parts),
             'absolute normalized ' + name + ' required')
    return value


def _digest(value, name):
    _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None
             and value != '0' * 64, 'nonzero lowercase SHA256 required for ' + name)
    return value


def _replace_once(source, before, after):
    _require(source.count(before) == 1, 'reviewed Java template fragment changed: ' + before[:70])
    return source.replace(before, after, 1)


def render_java(*, master_path, master_sha256, control_directory, campaign_id, jobs,
                behavior_digests):
    """Return Baseline1CampaignPlan.java, bound to actual master bytes and rows.

    Native P0/P1/P2/P4 jobs are calibration only; all other roles are BASELINE.
    Numeric/string CSV fields are accepted, but unknown or omitted columns fail.
    Every job still needs the existing ready snapshot and exclusive start claim.
    """
    master_path = _absolute(master_path, 'master path')
    control_directory = _absolute(control_directory, 'control directory')
    _digest(master_sha256, 'master plan')
    _require(isinstance(campaign_id, str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', campaign_id),
             'explicit safe campaign id required')
    _require(isinstance(behavior_digests, dict) and set(behavior_digests) == set(NATIVE_GROUPS),
             'explicit P0/P1/P2/P4 behavior digests required')
    for group, value in behavior_digests.items():
        _digest(value, group + ' offline behavior')
    _require(isinstance(jobs, (list, tuple)) and 0 < len(jobs) <= 2147483647, 'explicit nonempty Java jobs required')
    normalized = []
    seen = {key: set() for key in ('run_index', 'experiment_seed', 'actor_sampling_seed', 'event_bundle')}
    numeric = {'ordinal', 'actor_seed', 'run_index', 'experiment_seed', 'actor_sampling_seed',
               'target_training_update', 'checkpoint_update'}
    for index, job in enumerate(jobs):
        _require(isinstance(job, dict) and set(job) == set(COLUMNS), 'exact master columns required')
        values = {}
        for key in COLUMNS:
            value = job[key]
            _require(type(value) in (str, int), 'invalid CSV field type: ' + key)
            value = str(value)
            _require(value and value.strip() == value and
                     not any(character in value for character in (',', '"', '\r', '\n', '\t')),
                     'unsafe CSV field: ' + key)
            if key in numeric:
                _require(re.fullmatch('0|[1-9][0-9]*', value), 'invalid numeric CSV field: ' + key)
                limit = 2147483647 if key in ('ordinal', 'target_training_update', 'checkpoint_update') else 9223372036854775807
                _require(int(value) <= limit, 'numeric CSV field overflows Java: ' + key)
                if key in ('actor_seed', 'run_index', 'experiment_seed', 'actor_sampling_seed'):
                    _require(int(value) > 0, 'positive CSV field required: ' + key)
            values[key] = value
        _require(int(values['ordinal']) == index, 'master ordinal must be contiguous from zero')
        _require(values['group'] in GROUPS and values['role'] in ROLES, 'unknown Baseline 1 group/role')
        native = values['group'] in NATIVE_GROUPS
        _require((native and values['role'] == 'calibration') or
                 (not native and values['role'] in ROLES - {'calibration'}),
                 'native/calibration and BASELINE role combination differs')
        for key in ('event_directory', 'event_bundle', 'python_output'):
            _absolute(values[key], key)
        _require(Path(values['event_bundle']).parent == Path(values['event_directory']),
                 'event bundle must be a direct child of event directory')
        for key, observed in seen.items():
            _require(values[key] not in observed, 'duplicate Java master ' + key)
            observed.add(values[key])
        normalized.append(values)

    master_bytes = Path(master_path).read_bytes()
    _require(hashlib.sha256(master_bytes).hexdigest() == master_sha256, 'master actual byte SHA256 differs')
    master_text = master_bytes.decode('utf-8')
    _require('"' not in master_text and '\r' not in master_text.replace('\r\n', ''), 'unsupported CSV quoting/newline')
    lines = re.split(r'\r?\n', master_text)
    if lines and lines[-1] == '':
        lines.pop()
    _require(len(lines) == len(jobs) + 1 and all(lines), 'master must contain exactly the frozen job count')
    reader = csv.DictReader(io.StringIO(master_text, newline=''))
    _require(tuple(reader.fieldnames or ()) == COLUMNS and list(reader) == normalized,
             'frozen CSV differs from supplied Java jobs')

    source = TEMPLATE.read_text()
    source = source.replace('H2CampaignPlan', 'Baseline1CampaignPlan')
    source = source.replace('H2 one-click STOP:', 'Baseline 1 one-click STOP:')
    source = _replace_once(source, 'private static final int JOB_COUNT = 3744;',
                           'private static final int JOB_COUNT = ' + str(len(jobs)) + ';')
    source = _replace_once(source, '"master must contain exactly 3744 jobs"',
                           '"master must contain exactly " + JOB_COUNT + " jobs"')
    source = _replace_once(source,
        'require(values[2].equals("A") || values[2].equals("B") || values[2].equals("C"), "invalid group");',
        'require(values[2].equals("P0") || values[2].equals("P1") || values[2].equals("P2")\n'
        '                || values[2].equals("P4") || values[2].equals("BASELINE"), "invalid group");')
    source = _replace_once(source,
        'require(values[3].equals("train") || values[3].equals("monitor") || values[3].equals("final_blind"), "invalid role");',
        'require(values[3].equals("calibration") || values[3].equals("t0") || values[3].equals("train")\n'
        '                || values[3].equals("monitor") || values[3].equals("final"), "invalid role");\n'
        '            require(values[2].startsWith("P") == values[3].equals("calibration"),\n'
        '                "native/calibration and BASELINE role combination differs");')
    source = source.replace('stage8-one-click-ready-v1', 'baseline1-one-click-ready-v1')
    source = source.replace('stage8-one-click-java-start-v1', 'baseline1-one-click-java-start-v1')
    accessors = '\n'.join([
        '',
        '    public static int dispatchPolicy(int index) {',
        '        String group = row(index).cells[2];',
        '        return group.equals("BASELINE") ? 3 : Integer.parseInt(group.substring(1));',
        '    }',
        '    public static boolean offlineEnabled(int index) { return row(index).cells[2].startsWith("P"); }',
        '    public static String offlineSha256(int index) {',
        '        String group = row(index).cells[2];',
        *['        if (group.equals("' + group + '")) return "' + behavior_digests[group] + '";'
          for group in NATIVE_GROUPS],
        '        return "' + '0' * 64 + '";',
        '    }',
        '    public static boolean eventRecordingEnabled(int index) { return offlineEnabled(index); }',
    ])
    needle = '    public static String eventRoot(int index) { return row(index).cells[9]; }'
    source = _replace_once(source, needle, needle + '\n' + accessors)
    replacements = dict(MASTER_PLAN_PATH=master_path, MASTER_PLAN_SHA256=master_sha256,
                        CONTROL_DIRECTORY=control_directory, CAMPAIGN_ID=campaign_id)
    for key, value in replacements.items():
        source = _replace_once(source, '@@' + key + '@@', json.dumps(value, ensure_ascii=True)[1:-1])
    _require(re.search(r'@@[A-Z_]+@@', source) is None, 'unrendered Java configuration placeholder')
    return source
