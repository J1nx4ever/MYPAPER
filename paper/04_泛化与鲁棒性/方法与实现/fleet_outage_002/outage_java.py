"""Bind every Java Freeform row to its frozen paired fleet roster."""
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from outage_plan import (CAMPAIGN, CONTROL, COUNT, PLAN, SCENARIOS, SLOTS, SUITE, previous,
                         reference, require, validate_jobs)
from java_helper import master_bytes, scenario_table_bytes, verify_java as verify_nominal_java
from baseline1_v1.java_plan import _replace_once


def render(jobs, master, scenarios):
    validate_jobs(jobs)
    require(master.read_bytes() == master_bytes(jobs) and scenarios.read_bytes() == scenario_table_bytes(jobs),
            'fleet comparison Java tables differ from the canonical validation jobs')
    prior = previous.load_suite()
    source = Path(prior['java_helper']['path']).read_text(encoding='utf-8')
    new_master, new_scene = reference(master), reference(scenarios)
    pairs = [(prior['master_plan'][key], new_master[key]) for key in ('path', 'sha256')]
    pairs += [(prior['scenario_table'][key], new_scene[key]) for key in ('path', 'sha256')]
    old_launcher = reference(previous.PLAN/'launch.py')
    launcher = reference(PLAN/'launch.py')
    pairs += [(str(previous.SUITE), str(SUITE)), (str(previous.PLAN/'launch.py'), str(PLAN/'launch.py')),
              (old_launcher['sha256'], launcher['sha256']),
              (str(previous.CONTROL), str(CONTROL)), (previous.CAMPAIGN, CAMPAIGN)]
    for old, new in pairs:
        source = _replace_once(source, json.dumps(old), json.dumps(new))
    source = _replace_once(source, f'private static final int JOB_COUNT = {previous.COUNT};',
                           f'private static final int JOB_COUNT = {COUNT};\n'
                           f'    private static final int ROSTERS_PER_SLOT = {len(SCENARIOS)};')
    source = _replace_once(source, '''                    require(environmentSeeds.add(row.seed), "environment seed repeats a fleet outage slot");
                    require(samplingSeeds.add(positive(values[8], "actor sampling seed")),
                        "Actor sampling seed repeats a fleet outage slot");''', '''                    final int slotStart = i - i % ROSTERS_PER_SLOT;
                    final long actorSample = positive(values[8], "actor sampling seed");
                    if (slotStart == i) {
                        require(environmentSeeds.add(row.seed), "environment seed repeats a fleet outage slot");
                        require(samplingSeeds.add(actorSample), "Actor sampling seed repeats a fleet outage slot");
                    } else {
                        require(row.seed == rows[slotStart].seed
                            && values[8].equals(rows[slotStart].cells[8]), "paired seeds differ inside fleet slot");
                    }''')
    masks = ', '.join(str(job['fleet']['disabled_mask']) for job in jobs)
    ids = ', '.join(json.dumps(job['scene_id']) for job in jobs)
    return _replace_once(source, '    public static int fleetDisabledTruckMask(int index) { row(index); return 292; }',
                         f'''    private static final int[] FLEET_MASK_BY_ORDINAL = {{{masks}}};
    private static final String[] FLEET_SCENE_BY_ORDINAL = {{{ids}}};

    public static int fleetDisabledTruckMask(int index) {{
        Row selected = row(index);
        require(scenarioForRun(selected.run).id.equals(FLEET_SCENE_BY_ORDINAL[selected.ordinal]),
            "fleet scenario differs from the ordered frozen row");
        return FLEET_MASK_BY_ORDINAL[selected.ordinal];
    }}''')


def verify_java(helper, scenarios, jobs):
    source = helper.read_text(encoding='utf-8')
    launcher = reference(PLAN/'launch.py')
    require(json.dumps(launcher['path']) in source and json.dumps(launcher['sha256']) in source
            and json.dumps(reference(previous.PLAN/'launch.py')['sha256']) not in source,
            'Java startup does not seal the new comparison launcher')
    proof = verify_nominal_java(helper, scenarios)
    masks = ', '.join(str(job['fleet']['disabled_mask']) for job in jobs)
    runs = ', '.join(f"{job['run_index']}L" for job in jobs)
    environments = ', '.join(f"{job['experiment_seed']}L" for job in jobs)
    actors = ', '.join(f"{job['actor_sampling_seed']}L" for job in jobs)
    source = f'''public final class FleetMaskProbe {{
    public static void main(String[] args) {{
        final int[] masks = {{{masks}}};
        final long[] runs = {{{runs}}}, environments = {{{environments}}}, actors = {{{actors}}};
        java.util.Set<Long> seeds = new java.util.HashSet<>();
        for (int i = 0; i < Baseline1CampaignPlan.jobCount(); i++) {{
            if (Baseline1CampaignPlan.fleetDisabledTruckMask(i) != masks[i]
                    || Baseline1CampaignPlan.runIndex(i) != runs[i]
                    || Baseline1CampaignPlan.environmentSeed(i) != environments[i]
                    || Baseline1CampaignPlan.actorSamplingSeed(i) != actors[i]) {{
                throw new AssertionError("invalid fleet mask or paired seed scope");
            }}
            seeds.add(environments[i]);
            seeds.add(actors[i]);
        }}
        if (seeds.size() != {SLOTS*2}) throw new AssertionError("incomplete fleet identity inventory");
        for (int bad : new int[]{{-1, {COUNT}}}) {{
            try {{
                Baseline1CampaignPlan.fleetDisabledTruckMask(bad);
                throw new AssertionError("invalid fleet index accepted");
            }} catch (IllegalStateException expected) {{ }}
        }}
    }}
}}'''
    with TemporaryDirectory(prefix='fleet-paired-mask-') as name:
        path = Path(name)
        probe = path/'FleetMaskProbe.java'
        probe.write_text(source, encoding='ascii')
        subprocess.run(['javac', '-encoding', 'UTF-8', '-d', str(path), str(helper), str(probe)], check=True)
        subprocess.run(['java', '-cp', str(path), 'FleetMaskProbe'], check=True)
    return dict(proof, launcher_fingerprint_checked=True,
                fleet_disabled_masks=list(dict.fromkeys(job['fleet']['disabled_mask'] for job in jobs)),
                fleet_mask_rows_checked=COUNT, invalid_fleet_indices_rejected=[-1,COUNT],
                distinct_seeds_checked=SLOTS*2)
