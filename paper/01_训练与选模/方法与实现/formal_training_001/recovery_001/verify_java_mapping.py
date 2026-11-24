"""Compile the recovery helper in /tmp and verify all remaining identities."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path('/home/jinx/Faith/experiments/lagrangian_ppo_v1')
PLAN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PLAN))


def verify(report_path):
    from prepare_recovery import render_java
    from business_graph_summary_campaign_v1.plan_io import reference, load_suite, write
    from business_graph_summary_campaign_v1.gate import check_sources, PROTECTED_SHA
    original_path = PLAN.parent / 'prepared_001/suite.json'
    original, _ = load_suite(original_path)
    check_sources(original['source_pins'])
    immutable = [reference(path) for path in (
        original_path, Path(original['master_plan']['path']), Path(original['java_helper']['path']),
        ROOT.parents[1] / 'python/episode_budget_ppo.py',
        Path(original['control_directory']) / 'STOP.properties')]
    control = Path(original['python_output']) / 'recovery_001/control'
    assert not control.exists()
    source = render_java(Path(original['java_helper']['path']).read_text(), old_suite=original,
        recovery_path=PLAN / 'recovery.json', control_directory=control, bridge_path=PLAN / 'bootstrap.py')
    frozen_helper = PLAN / 'Baseline1CampaignPlan.java'
    if frozen_helper.is_file():
        assert frozen_helper.read_text() == source
    probe = '''
public final class Probe {
    private static void require(boolean b) { if(!b) throw new AssertionError("mapping mismatch"); }
    public static void main(String[] args) {
        require(Baseline1CampaignPlan.jobCount()==511);
        require(Baseline1CampaignPlan.frozenPlanJobCount()==512);
        require(Baseline1CampaignPlan.resumeOffset()==1);
        for(int i=0;i<511;i++) {
            require(Baseline1CampaignPlan.dispatchPolicy(i)==3);
            require(Baseline1CampaignPlan.materialEnabled(i));
            require(!Baseline1CampaignPlan.offlineEnabled(i));
            require(!Baseline1CampaignPlan.eventRecordingEnabled(i));
            require(Baseline1CampaignPlan.stageId(i).equals("BG128"));
            System.out.println(i+"\\t"+Baseline1CampaignPlan.runIndex(i)+"\\t"+
                Baseline1CampaignPlan.environmentSeed(i)+"\\t"+
                Baseline1CampaignPlan.actorSamplingSeed(i)+"\\t"+
                Baseline1CampaignPlan.recordingId(i)+"\\t"+
                Baseline1CampaignPlan.phaseId(i)+"\\t"+
                Baseline1CampaignPlan.eventRoot(i)+"\\t"+
                Baseline1CampaignPlan.materialRoot(i));
        }
        boolean low=false, high=false, captured=false;
        try { Baseline1CampaignPlan.runIndex(-1); } catch(IllegalStateException expected) { low=true; }
        try { Baseline1CampaignPlan.runIndex(511); } catch(IllegalStateException expected) { high=true; }
        try {
            Baseline1CampaignPlan.awaitConfigSha256(9900001L,255314794623997L,
                "business-graph-summary-formal-seed1-20261118-001-s01_BG128_train_u0001",
                "@EVENTS@", "boundary-events-v1-native-rpc-runtime-scene-v1");
        } catch(IllegalStateException expected) {
            captured=expected.getMessage().contains("already captured original job");
        }
        require(low && high && captured);
    }
}
'''.replace('@EVENTS@', original['jobs'][0]['event_directory'])
    with tempfile.TemporaryDirectory(prefix='bg128-recovery-java-') as directory:
        folder = Path(directory)
        helper = folder / 'Baseline1CampaignPlan.java'
        helper.write_text(source)
        probe_path = folder / 'Probe.java'
        probe_path.write_text(probe)
        compile_source = frozen_helper if frozen_helper.is_file() else helper
        compiled = subprocess.run(['javac', '-d', directory, str(compile_source), str(probe_path)],
            capture_output=True, text=True, timeout=60)
        assert compiled.returncode == 0, compiled.stderr + compiled.stdout
        executed = subprocess.run(['java', '-cp', directory, 'Probe'],
            capture_output=True, text=True, timeout=60)
        assert executed.returncode == 0, executed.stderr + executed.stdout
    rows = [line.split('\t') for line in executed.stdout.splitlines()]
    assert len(rows) == 511
    for index, (values, job) in enumerate(zip(rows, original['jobs'][1:])):
        assert values == [str(index), str(job['run_index']), str(job['experiment_seed']),
            str(job['actor_sampling_seed']), job['recording_id'], job['phase_id'],
            job['event_directory'], job['material_event_directory']]
    assert all(reference(item['path']) == item for item in immutable)
    assert immutable[-2]['sha256'] == PROTECTED_SHA
    assert not control.exists()
    result = dict(status=('PASS_FROZEN_HELPER_READ_ONLY_MAPPING_ONLY' if frozen_helper.is_file()
                          else 'PASS_TMP_RENDER_AND_READ_ONLY_MAPPING_ONLY'),
        original_suite=reference(original_path), generated_helper_sha256=hashlib.sha256(source.encode()).hexdigest(),
        checks=dict(java_compiled=True, all_511_rows_exact_original_suffix=True,
            count_511_frozen_512_offset_1=True, out_of_range_rejected=True,
            already_captured_9900001_rejected_before_bridge=True,
            original_sources_helper_master_stop_preserved=True,
            protected_ppo_unchanged=True, recovery_control_absent=True),
        observed_first_run=9900002, observed_last_run=9900512,
        temporary_compile_only=True, descriptor_frozen=(PLAN / 'recovery.json').is_file(),
        compiled_frozen_helper=frozen_helper.is_file(),
        frozen_helper=reference(frozen_helper) if frozen_helper.is_file() else None, budget_api_calls=0,
        reconciliation_applied_by_this_check=False, controller_started=False, anylogic_started=False,
        optimizer_updates=0, protected_ppo_sha256=PROTECTED_SHA,
        javac_stderr=compiled.stderr, java_stderr=executed.stderr)
    write(report_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.report), ensure_ascii=False))


if __name__ == '__main__':
    main()
