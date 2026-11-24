"""Check the complete frozen Java mapping and controller boundaries offline."""
from collections import Counter
from pathlib import Path
import subprocess
import tempfile

from extension_common import (
    APPROVAL, CAMPAIGN, CORE, MATERIALS, OUTPUT, PREPARED, REPORT, SUITE, TOTAL,
    checked, load_package, reference, require,
)
from business_graph_summary_campaign_v1.plan_io import write
from business_graph_summary_recovery_v1.collection import MaterialPhaseCollector
from extension_budget import ExtensionLedger, budget_snapshot, check_collisions
from train_extension import verify_initial_checkpoint


def verify_java(suite, folder):
    probe = folder / 'Probe.java'
    probe.write_text('''
import java.nio.file.*;
public class Probe {
    public static void main(String[] args) throws Exception {
        var rows = Files.readAllLines(Path.of(args[0]));
        int count = rows.size() - 1;
        if (count != Baseline1CampaignPlan.jobCount() ||
            count != Baseline1CampaignPlan.frozenPlanJobCount() ||
            Baseline1CampaignPlan.resumeOffset() != 0) throw new AssertionError("job count");
        for (int i = 0; i < count; i++) {
            var row = rows.get(i + 1).split(",");
            if (Baseline1CampaignPlan.runIndex(i) != Long.parseLong(row[6]) ||
                Baseline1CampaignPlan.environmentSeed(i) != Long.parseLong(row[7]) ||
                Baseline1CampaignPlan.actorSamplingSeed(i) != Long.parseLong(row[8]) ||
                !Baseline1CampaignPlan.recordingId(i).equals(row[5]) ||
                !Baseline1CampaignPlan.eventRoot(i).equals(row[9]) ||
                !Baseline1CampaignPlan.materialRoot(i).equals(args[1] + "/" + row[5].substring(args[2].length() + 1)) ||
                !Baseline1CampaignPlan.stageId(i).equals("BG128") ||
                Baseline1CampaignPlan.dispatchPolicy(i) != 3 ||
                Baseline1CampaignPlan.eventRecordingEnabled(i) ||
                !Baseline1CampaignPlan.materialEnabled(i)) throw new AssertionError("row " + i);
        }
        System.out.println("PASS complete Java mapping " + count);
    }
}
''')
    subprocess.run(['javac', '-d', str(folder), str(checked(suite['java_helper'])), str(probe)], check=True)
    subprocess.run(['java', '-cp', str(folder), 'Probe', str(checked(suite['master_plan'])),
                    str(MATERIALS), CAMPAIGN], check=True)


def verify():
    import torch

    suite, phases = load_package()
    previous = budget_snapshot()
    check_collisions(suite, previous)
    torch.set_num_threads(suite['cpu_threads'])
    proof = verify_initial_checkpoint(suite['parent_checkpoint'], suite['learner_config'])
    with tempfile.TemporaryDirectory(prefix='bg128-extension-preflight-') as temp:
        d = Path(temp)
        verify_java(suite, d)
        ledger = ExtensionLedger(suite, suite_path=SUITE, approval_path=APPROVAL, path=d / 'ledger.sqlite')
        ledger.register()
        state = ledger.snapshot()
        require(Counter(row['status'] for row in state['jobs'].values()) == {'PLANNED': TOTAL}
                and state['campaigns'][CAMPAIGN]['commits'] == {}, 'temporary ledger registration differs')
        for phase, cfg in zip(suite['phases'], phases):
            MaterialPhaseCollector(cfg, checked(phase['phase_config']),
                                   publish_ready=lambda job, path: None,
                                   startup=dict(control_directory=suite['control_directory'],
                                                campaign_id=CAMPAIGN,
                                                master_plan_sha256=suite['master_plan']['sha256']))
        first, second = phases[0]['jobs'][:2]
        try:
            ledger.claim(second)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError('planned predecessor did not block slot')
        ledger.claim(first)
        require(ledger.job(first)['status'] == 'CLAIMED', 'exact first slot claim failed')
        ledger.unknown(first, 'isolated verification of the UNK transition')
        require(ledger.job(first)['status'] == 'UNKNOWN', 'unknown slot evidence missing')
    require(not OUTPUT.exists() and not MATERIALS.exists()
            and not (PREPARED / 'launch').exists(), 'preflight created live output')
    write(REPORT / 'preparation_validation.json',
          dict(status='PASS_OFFLINE_PREPARATION', suite=reference(SUITE), checked_java_rows=TOTAL,
               checked_phase_collectors=len(phases), preceding_cap=previous['maximum_total_shifts'],
               original_checkpoint_restore=proof, protected_core=reference(CORE),
               physical_shifts_started=0, optimizer_updates=0))
    print('PASS frozen continuation, Java, seed pairing and budget gates')


if __name__ == '__main__':
    verify()
