"""Adapt the sealed nominal ready/claim renderer to one deterministic row."""
import json

from dispatch_plan import CAMPAIGN, CONTROL, PLAN, POLICY, SUITE, reference, require, smoke, validate_jobs
from java_helper import master_bytes, render as render_smoke, scenario_table_bytes, verify_java
from baseline1_v1.java_plan import _replace_once


def render(jobs, master, scenarios):
    validate_jobs(jobs)
    require(master.read_bytes() == master_bytes(jobs) and scenarios.read_bytes() == scenario_table_bytes(jobs),
            'DISPATCH Java data differ from the canonical validation job')
    validated_smoke = smoke.load_suite(check_sources=False)
    smoke_master, smoke_scenarios = (smoke.PREPARED/name for name in ('master_plan.csv', 'scenario_table.csv'))
    source = render_smoke(validated_smoke['jobs'], smoke_master, smoke_scenarios,
                          suite_path=SUITE, bridge_path=PLAN/'launch.py')
    bindings = dict(MASTER_PLAN_PATH=(str(smoke_master), str(master)),
                    MASTER_PLAN_SHA256=(reference(smoke_master)['sha256'], reference(master)['sha256']),
                    SCENARIO_TABLE_PATH=(str(smoke_scenarios), str(scenarios)),
                    SCENARIO_TABLE_SHA256=(reference(smoke_scenarios)['sha256'], reference(scenarios)['sha256']),
                    CONTROL_DIRECTORY=(str(smoke.CONTROL), str(CONTROL)), CAMPAIGN_ID=(smoke.CAMPAIGN, CAMPAIGN))
    for name, (old, new) in bindings.items():
        source = _replace_once(source, f'private static final String {name} = {json.dumps(old)};',
                              f'private static final String {name} = {json.dumps(new)};')
    source = _replace_once(source, 'private static final int JOB_COUNT = 4;', 'private static final int JOB_COUNT = 1;')
    for index, field in [(1, 'actor_seed'), (8, 'actor_sampling_seed')]:
        start = f'            if (ordinal == JOB_COUNT - 1) positive(values[{index}], "{field}");\n            else require'
        source = _replace_once(source, start, '            require')
    source = _replace_once(source, '''                    require(i == 0 || row.seed == rows[0].seed,
                        "smoke rows must share this campaign's fresh environment seed");
''', '')
    return _replace_once(source,
        'public static String phaseId(int index) { return recordingId(index).substring(CAMPAIGN_ID.length() + 1); }',
        f'public static String phaseId(int index) {{ row(index); return "{POLICY}"; }}')
