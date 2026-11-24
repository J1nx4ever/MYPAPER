"""Cross-check a prepared package through Java Freeform getters without starting a shift."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import subprocess
import tempfile

from resilience_campaign import load_suite, reference


PROBE = """public class FreeformProbe extends Object {
    public static void main(String[] args) {
        for (int i = 0; i < Baseline1CampaignPlan.jobCount(); ++i) {
            long run = Baseline1CampaignPlan.runIndex(i);
            StringBuilder line = new StringBuilder();
            Object[] values = {run, Baseline1CampaignPlan.recordingId(i),
                Baseline1CampaignPlan.environmentSeed(i), Baseline1CampaignPlan.actorSamplingSeed(i),
                Baseline1CampaignPlan.eventRoot(i), Baseline1CampaignPlan.fleetDisabledTruckMask(i),
                Baseline1CampaignPlan.faultTruckId(i), Baseline1CampaignPlan.faultTriggerSeconds(i),
                Baseline1CampaignPlan.oreTarget(i), Baseline1CampaignPlan.wasteTarget(i),
                Baseline1CampaignPlan.wasteStop(i), Baseline1CampaignPlan.materialEnabled(i),
                Baseline1CampaignPlan.eventRecordingEnabled(i), Baseline1CampaignPlan.offlineEnabled(i),
                Baseline1CampaignPlan.dispatchPolicy(i),
                Baseline1CampaignPlan.roadSpeedLimit(run, "intersection1", "Crusher", 10.0)};
            for (Object v : values) line.append(v).append(',');
            for (int s = 0; s < 4; ++s)
                line.append(Baseline1CampaignPlan.loadIsOre(run, s)).append(',')
                    .append(Baseline1CampaignPlan.loadGrade(run, s)).append(',')
                    .append(Baseline1CampaignPlan.loadGradeHalfRange(run, s)).append(',');
            System.out.println(line.toString());
        }
    }
}
"""


def verify_package(path):
    suite = load_suite(path)
    jobs = suite['jobs']
    with tempfile.TemporaryDirectory() as temp:
        probe = Path(temp)/'FreeformProbe.java'
        probe.write_text(PROBE)
        subprocess.run(['javac','-d',temp, suite['java_helper']['path'],str(probe)],check=True)
        output = subprocess.run(['java','-cp',temp,'FreeformProbe'],check=True,capture_output=True,text=True)
    with open(suite['identities']['path'],newline='') as f:
        identities = list(csv.DictReader(f))
    for j, row, identity in zip(jobs,csv.reader(output.stdout.splitlines()),identities,strict=True):
        for key in identity:
            if identity[key] != str(j[key]):
                raise ValueError('identity table differs for '+key)
        actual = row[:16]
        expected = [j['run_index'],j['recording_id'],j['experiment_seed'],j['actor_sampling_seed'],
                    j['event_directory'],511 ^ j['initial_mask'],j['fault_truck_id'],j['fault_trigger_seconds'],
                    j['ore_target'],j['waste_target'],j['waste_stop'],'false','false','false',3,
                    j['expected_rpc_config']['config.runtime_scene.road.6.speedLimit_mps']]
        for raw, value in zip(actual,expected,strict=True):
            if (float(raw) != value if isinstance(value,float) else raw != str(value)):
                raise ValueError('Freeform value differs in run '+str(j['run_index']))
        for s, role in enumerate(j['assignment']):
            ore, grade, half = row[16+3*s:19+3*s]
            if (ore != str(role!='W').lower() or float(grade)!=j['grades'][s] or float(half)!=j['half_ranges'][s]):
                raise ValueError('Java scene differs in run '+str(j['run_index']))
    return dict(suite_sha256=reference(path)['sha256'], verified_physical_identities=len(jobs),
                verified_java_freeform_rows=len(jobs), first_run=jobs[0]['run_index'],last_run=jobs[-1]['run_index'],
                phase_counts=dict(Counter(j['split'] if j['role']=='eval' else j['arm'] for j in jobs)),
                budget=suite['budget_at_preparation'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True)
    print(json.dumps(verify_package(p.parse_args().suite),indent=2))
