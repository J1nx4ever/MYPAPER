"""Adapt the sealed one-click Java renderer to the 100 paired nominal slots."""
import json

from paired_plan import CAMPAIGN, CONTROL, COUNT, PLAN, SUITE, reference, require, smoke, validate_jobs
from java_helper import master_bytes, render as render_smoke, scenario_table_bytes, verify_java
from baseline1_v1.java_plan import _replace_once


def render(jobs, master, scenarios):
    validate_jobs(jobs)
    require(master.read_bytes() == master_bytes(jobs) and scenarios.read_bytes() == scenario_table_bytes(jobs),
            'paired Java data differ from the canonical validation jobs')
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
    source = _replace_once(source, 'private static final int JOB_COUNT = 4;',
                          f'private static final int JOB_COUNT = {COUNT};')
    require(source.count('ordinal == JOB_COUNT - 1') == 2, 'reviewed BG128 seed checks differ')
    source = source.replace('ordinal == JOB_COUNT - 1', 'ordinal % 4 == 3')
    source = _replace_once(source, '                for (int i = 0; i < JOB_COUNT; i++) {', '''
                java.util.Set<Long> environmentSeeds = new java.util.HashSet<>();
                java.util.Set<Long> samplingSeeds = new java.util.HashSet<>();
                for (int i = 0; i < JOB_COUNT; i++) {''')
    source = _replace_once(source, '''                    require(i == 0 || row.seed == rows[0].seed,
                        "smoke rows must share this campaign's fresh environment seed");''', '''
                    if (i % 4 == 0) require(environmentSeeds.add(row.seed), "environment seed repeats a paired slot");
                    else require(row.seed == rows[i - 1].seed, "the four policies must share the slot's seed");
                    if (i % 4 == 3) require(samplingSeeds.add(positive(values[8], "actor sampling seed")),
                        "BG128 sampling seed repeats a paired slot");''')
    return _replace_once(source, '                cached = new Plan(rows, byRun);', '''
                require(java.util.Collections.disjoint(environmentSeeds, samplingSeeds), "environment and sampling seeds overlap");
                cached = new Plan(rows, byRun);''')
