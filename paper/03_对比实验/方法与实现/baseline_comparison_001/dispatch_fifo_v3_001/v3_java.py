"""Reuse the nominal Java handshake for the deterministic V3 roster."""
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from v3_plan import (CAMPAIGN, CONTROL, COUNT, PLAN, POLICY, SUITE, reference,
                     require, smoke, source_contract, validate_jobs)
from java_helper import master_bytes, render as render_smoke, scenario_table_bytes, verify_java as verify_nominal_java
from baseline1_v1.java_plan import _replace_once


def render(jobs, master, scenarios):
    _, anchors = source_contract()
    validate_jobs(jobs, anchors)
    require(master.read_bytes() == master_bytes(jobs) and scenarios.read_bytes() == scenario_table_bytes(jobs),
            'FIFO DISPATCH Java tables differ from the frozen roster')
    validated = smoke.load_suite(check_sources=False)
    smoke_master, smoke_scenarios = (smoke.PREPARED/name for name in ('master_plan.csv', 'scenario_table.csv'))
    source = render_smoke(validated['jobs'], smoke_master, smoke_scenarios,
                          suite_path=SUITE, bridge_path=PLAN/'launch.py')
    bindings = dict(MASTER_PLAN_PATH=(str(smoke_master),str(master)),
                    MASTER_PLAN_SHA256=(reference(smoke_master)['sha256'],reference(master)['sha256']),
                    SCENARIO_TABLE_PATH=(str(smoke_scenarios),str(scenarios)),
                    SCENARIO_TABLE_SHA256=(reference(smoke_scenarios)['sha256'],reference(scenarios)['sha256']),
                    CONTROL_DIRECTORY=(str(smoke.CONTROL),str(CONTROL)), CAMPAIGN_ID=(smoke.CAMPAIGN,CAMPAIGN))
    for name, (old,new) in bindings.items():
        source = _replace_once(source, f'private static final String {name} = {json.dumps(old)};',
                               f'private static final String {name} = {json.dumps(new)};')
    source = _replace_once(source, 'private static final int JOB_COUNT = 4;',
                           f'private static final int JOB_COUNT = {COUNT};')
    for index, field in ((1,'actor_seed'), (8,'actor_sampling_seed')):
        start = f'            if (ordinal == JOB_COUNT - 1) positive(values[{index}], "{field}");\n            else require'
        source = _replace_once(source, start, '            require')
    source = _replace_once(source, '                for (int i = 0; i < JOB_COUNT; i++) {', '''
                java.util.Set<Long> environmentSeeds = new java.util.HashSet<>();
                for (int i = 0; i < JOB_COUNT; i++) {''')
    source = _replace_once(source, '''                    require(i == 0 || row.seed == rows[0].seed,
                        "smoke rows must share this campaign's fresh environment seed");''', '''
                    require(environmentSeeds.add(row.seed), "environment seed repeats a FIFO DISPATCH slot");''')
    return _replace_once(source,
        'public static String phaseId(int index) { return recordingId(index).substring(CAMPAIGN_ID.length() + 1); }',
        f'public static String phaseId(int index) {{ row(index); return "{POLICY}"; }}')


def verify_java(helper, scenarios, jobs):
    source = helper.read_text(encoding='utf-8')
    launcher = reference(PLAN/'launch.py')
    require(json.dumps(launcher['path']) in source and json.dumps(launcher['sha256']) in source,
            'Java startup does not seal the FIFO DISPATCH launcher')
    proof = verify_nominal_java(helper, scenarios)
    runs = ', '.join(f"{job['run_index']}L" for job in jobs)
    seeds = ', '.join(f"{job['experiment_seed']}L" for job in jobs)
    probe_source = f'''public final class V3PairProbe {{
    public static void main(String[] args) {{
        final long[] runs = {{{runs}}}, seeds = {{{seeds}}};
        java.util.Set<Long> unique = new java.util.HashSet<>();
        for (int i = 0; i < Baseline1CampaignPlan.jobCount(); i++) {{
            if (Baseline1CampaignPlan.runIndex(i) != runs[i]
                    || Baseline1CampaignPlan.environmentSeed(i) != seeds[i]
                    || Baseline1CampaignPlan.actorSamplingSeed(i) != 0L
                    || !unique.add(seeds[i])
                    || !Baseline1CampaignPlan.phaseId(i).equals("{POLICY}")) {{
                throw new AssertionError("wrong FIFO DISPATCH pair identity");
            }}
        }}
        if (unique.size() != {COUNT}) throw new AssertionError("incomplete FIFO DISPATCH environment roster");
    }}
}}'''
    with TemporaryDirectory(prefix='v3-pair-probe-') as name:
        path = Path(name)
        probe = path/'V3PairProbe.java'
        probe.write_text(probe_source, encoding='ascii')
        subprocess.run(['javac','-encoding','UTF-8','-d',str(path),str(helper),str(probe)], check=True)
        subprocess.run(['java','-cp',str(path),'V3PairProbe'], check=True)
    return dict(proof, launcher_fingerprint_checked=True, paired_environment_rows_checked=COUNT,
                no_actor_sampling_rows_checked=COUNT)
