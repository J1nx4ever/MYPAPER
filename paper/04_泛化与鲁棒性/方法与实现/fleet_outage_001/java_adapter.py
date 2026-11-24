"""Reuse nominal Java readiness, adding the frozen fleet outage mask."""
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from fleet_plan import CAMPAIGN, CONTROL, COUNT, DISABLED_MASK, PLAN, SUITE, reference, require, smoke, validate_jobs
from java_helper import master_bytes, render as render_smoke, scenario_table_bytes, verify_java as verify_nominal_java
from baseline1_v1.java_plan import _replace_once


def render(jobs, master, scenarios):
    validate_jobs(jobs)
    require(master.read_bytes() == master_bytes(jobs) and scenarios.read_bytes() == scenario_table_bytes(jobs),
            'fleet outage Java tables differ from the canonical validation jobs')
    validated = smoke.load_suite(check_sources=False)
    smoke_master, smoke_scenarios = (smoke.PREPARED/name for name in ('master_plan.csv', 'scenario_table.csv'))
    source = render_smoke(validated['jobs'], smoke_master, smoke_scenarios,
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
    source = _replace_once(source, '''            if (ordinal == JOB_COUNT - 1) positive(values[1], "actor_seed");
            else require(values[1].equals("0"), "classical policy cannot have an Actor initialization");''',
                          '            positive(values[1], "actor_seed");')
    source = _replace_once(source, '''            if (ordinal == JOB_COUNT - 1) positive(values[8], "actor_sampling_seed");
            else require(values[8].equals("0"), "classical policy must not consume a sampling seed");''',
                          '            positive(values[8], "actor_sampling_seed");')
    source = _replace_once(source, '                for (int i = 0; i < JOB_COUNT; i++) {', '''
                java.util.Set<Long> environmentSeeds = new java.util.HashSet<>();
                java.util.Set<Long> samplingSeeds = new java.util.HashSet<>();
                for (int i = 0; i < JOB_COUNT; i++) {''')
    source = _replace_once(source, '''                    require(i == 0 || row.seed == rows[0].seed,
                        "smoke rows must share this campaign's fresh environment seed");''', '''
                    require(environmentSeeds.add(row.seed), "environment seed repeats a fleet outage slot");
                    require(samplingSeeds.add(positive(values[8], "actor sampling seed")),
                        "Actor sampling seed repeats a fleet outage slot");''')
    source = _replace_once(source, '                cached = new Plan(rows, byRun);', '''
                require(java.util.Collections.disjoint(environmentSeeds, samplingSeeds), "environment and sampling seeds overlap");
                cached = new Plan(rows, byRun);''')
    anchor = '    public static int dispatchPolicy(int index) { row(index); return 3; }'
    return _replace_once(source, anchor, anchor + '\n'
                         f'    public static int fleetDisabledTruckMask(int index) {{ row(index); return {DISABLED_MASK}; }}')


def verify_java(helper, scenarios):
    proof = verify_nominal_java(helper, scenarios)
    source = f'''public final class FleetMaskProbe {{
    public static void main(String[] args) {{
        java.util.Set<Long> seeds = new java.util.HashSet<>();
        for (int i = 0; i < Baseline1CampaignPlan.jobCount(); i++) {{
            if (Baseline1CampaignPlan.fleetDisabledTruckMask(i) != {DISABLED_MASK}
                    || !seeds.add(Baseline1CampaignPlan.environmentSeed(i))
                    || !seeds.add(Baseline1CampaignPlan.actorSamplingSeed(i))) {{
                throw new AssertionError("invalid fleet mask or seed scope");
            }}
        }}
        if (seeds.size() != {COUNT*2}) throw new AssertionError("incomplete fleet identity inventory");
        for (int bad : new int[]{{-1, {COUNT}}}) {{
            try {{
                Baseline1CampaignPlan.fleetDisabledTruckMask(bad);
                throw new AssertionError("invalid fleet index accepted");
            }} catch (IllegalStateException expected) {{ }}
        }}
    }}
}}'''
    with TemporaryDirectory(prefix='fleet-mask-probe-') as name:
        path = Path(name)
        probe = path/'FleetMaskProbe.java'
        probe.write_text(source, encoding='ascii')
        subprocess.run(['javac', '-encoding', 'UTF-8', '-d', str(path), str(helper), str(probe)], check=True)
        subprocess.run(['java', '-cp', str(path), 'FleetMaskProbe'], check=True)
    return dict(proof, fleet_disabled_mask=DISABLED_MASK, fleet_mask_rows_checked=COUNT,
                invalid_fleet_indices_rejected=[-1,COUNT], distinct_seeds_checked=COUNT*2)
