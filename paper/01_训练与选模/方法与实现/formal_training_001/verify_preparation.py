"""Read-only frozen-plan preflight and Java compilation/accessor verification."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path('/home/jinx/Faith/experiments/lagrangian_ppo_v1')
sys.path.insert(0, str(ROOT))


def verify(suite_path, report_path):
    from business_graph_summary_campaign_v1.plan_io import reference, write, sha
    from business_graph_summary_campaign_v1.gate import preflight, PROTECTED_SHA
    from business_graph_summary_campaign_v1.budget import (
        GLOBAL_PATH, preceding_reference, budget_snapshot)
    suite_path = Path(suite_path).resolve()
    before_budget = preceding_reference()
    before_global = reference(GLOBAL_PATH) if GLOBAL_PATH.exists() else None
    suite, phases, approval = preflight(suite_path)
    assert approval is None
    output = Path(suite['python_output'])
    material = output.with_name(suite['campaign_id'] + '_materials')
    assert not output.exists() and not material.exists()
    assert sha(ROOT.parents[1] / 'python/episode_budget_ppo.py') == PROTECTED_SHA
    with Path(suite['master_plan']['path']).open(newline='') as stream:
        master_rows = list(csv.DictReader(stream))
    assert len(master_rows) == len(suite['jobs']) == 512
    probe = '''
public final class Probe {
    public static void main(String[] args) {
        if (Baseline1CampaignPlan.jobCount() != 512 ||
            Baseline1CampaignPlan.frozenPlanJobCount() != 512 ||
            Baseline1CampaignPlan.resumeOffset() != 0 ||
            Baseline1CampaignPlan.preparationOnly()) throw new AssertionError("counts");
        for (int i=0;i<512;i++) {
            if (Baseline1CampaignPlan.dispatchPolicy(i) != 3 ||
                Baseline1CampaignPlan.offlineEnabled(i) ||
                Baseline1CampaignPlan.eventRecordingEnabled(i) ||
                !Baseline1CampaignPlan.materialEnabled(i) ||
                Baseline1CampaignPlan.waitWeight(i) != .2 ||
                !Baseline1CampaignPlan.stageId(i).equals("BG128")) throw new AssertionError("accessors");
            System.out.println(i+"\\t"+Baseline1CampaignPlan.runIndex(i)+"\\t"+
                Baseline1CampaignPlan.environmentSeed(i)+"\\t"+
                Baseline1CampaignPlan.actorSamplingSeed(i)+"\\t"+
                Baseline1CampaignPlan.recordingId(i)+"\\t"+
                Baseline1CampaignPlan.phaseId(i)+"\\t"+
                Baseline1CampaignPlan.eventRoot(i)+"\\t"+
                Baseline1CampaignPlan.materialRoot(i));
        }
    }
}
'''
    with tempfile.TemporaryDirectory(prefix='bg128-java-plan-') as folder:
        temporary = Path(folder)
        probe_path = temporary / 'Probe.java'
        probe_path.write_text(probe)
        compiled = subprocess.run(['javac', '-d', str(temporary),
            suite['java_helper']['path'], str(probe_path)], capture_output=True, text=True,
            timeout=60, check=True)
        observed = subprocess.run(['java', '-cp', str(temporary), 'Probe'],
            capture_output=True, text=True, timeout=60)
        if observed.returncode:
            raise ValueError('read-only Java probe failed: ' + observed.stderr + observed.stdout)
    rows = [line.split('\t') for line in observed.stdout.splitlines()]
    assert len(rows) == 512
    for index, (values, job) in enumerate(zip(rows, suite['jobs'])):
        assert values == [str(index), str(job['run_index']), str(job['experiment_seed']),
            str(job['actor_sampling_seed']), job['recording_id'], job['phase_id'], job['event_directory'],
            job['material_event_directory']]
    assert preceding_reference() == before_budget
    assert (reference(GLOBAL_PATH) if GLOBAL_PATH.exists() else None) == before_global
    assert not output.exists() and not material.exists()
    assert sha(ROOT.parents[1] / 'python/episode_budget_ppo.py') == PROTECTED_SHA
    snapshot = budget_snapshot()
    result = dict(status='PASS_PREPARED_NOT_REGISTERED_NOT_STARTED', suite=reference(suite_path),
        checks=dict(preflight=True, sixteen_phases=len(phases) == 16,
            java_compiled=True, all_512_read_only_accessor_rows_match=True,
            source_and_config_byte_sha_verified=True, merged_ledger_unchanged=True,
            independent_budget_unchanged=True, run_output_absent=True,
            material_output_absent=True, protected_ppo_unchanged=True),
        physical_shifts=512, training_updates=16, training_seed_count=1,
        merged_reserved_before=snapshot['maximum_total_shifts'],
        merged_reserved_if_registered=suite['maximum_total_if_approved'],
        javac_stderr=compiled.stderr, java_stderr=observed.stderr,
        stage_accessor=dict(observed='BG128', master_role='train',
            interpretation='inherited_representation_metadata_separate_from_master_role',
            initial_probe_expected_train_incorrectly=True,
            correction='test expectation corrected; frozen preparation and helper unchanged',
            runtime_stage_fault_found=False),
        java_probe_calls_await_config=False, listener_started=False, anylogic_started=False,
        budget_registered=False, optimizer_updates=0, protected_ppo_sha256=PROTECTED_SHA)
    write(report_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.suite, args.report), ensure_ascii=False))


if __name__ == '__main__':
    main()
