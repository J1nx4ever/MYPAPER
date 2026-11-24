"""Check the sealed controller boundary on temporary paths without a listener."""
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest.mock import patch

import torch

import discovery_budget
import run_discovery
from discovery_common import (CORE, COUNT, FIRST_RUN, OLD, OUTPUT, PREPARED, REPORT, SUITE,
    check_fresh_jobs, check_package, checked, previous_budget, read, reference, require, write)
from business_graph_summary_evaluation_v1.runtime import _load_model, _publisher
from business_graph_summary_recovery_v1.collection import MaterialPhaseCollector


def check_java(cfg, helper, temp):
    require(shutil.which('javac') and shutil.which('java'),'local JDK required')
    j = temp/'Probe.java'
    j.write_text('''
import java.nio.file.*;
public class Probe {
    public static void main(String[] a) throws Exception {
        if (Baseline1CampaignPlan.jobCount()!=100 || Baseline1CampaignPlan.resumeOffset()!=0)
            throw new AssertionError("count/offset");
        var rows = Files.readAllLines(Path.of(a[0]));
        for (int i=0;i<100;i++) {
            var c = rows.get(i+1).split(",");
            if (Baseline1CampaignPlan.runIndex(i)!=Long.parseLong(c[6]) ||
                Baseline1CampaignPlan.environmentSeed(i)!=Long.parseLong(c[7]) ||
                Baseline1CampaignPlan.actorSamplingSeed(i)!=Long.parseLong(c[8]) ||
                !Baseline1CampaignPlan.recordingId(i).equals(a[1]) ||
                !Baseline1CampaignPlan.materialRoot(i).equals(a[2]) ||
                Baseline1CampaignPlan.dispatchPolicy(i)!=3 ||
                Baseline1CampaignPlan.eventRecordingEnabled(i) ||
                !Baseline1CampaignPlan.materialEnabled(i))
                throw new AssertionError(i);
        }
        System.out.println("PASS all Java identities");
    }
}
''')
    subprocess.run(['javac','-d',str(temp),str(helper),str(j)],check=True)
    subprocess.run(['java','-cp',str(temp),'Probe',str(PREPARED/'master_plan.csv'),cfg['recording_id'],
                    cfg['jobs'][0]['material_event_directory']],check=True)


def main():
    suite,cfg = check_package()
    prior = previous_budget()
    check_fresh_jobs(cfg['jobs'],prior)
    require(prior['maximum_total_shifts']==21699 and all(
        prior['jobs'][str(i)]['status']=='COMPLETE' for i in range(9980001,9980005)), 'prior four-slot ledger omitted')
    torch.set_num_threads(2)
    model = _load_model(suite)
    require(model.config['ppo']['actor_lr']==.0003 and model.update_index==13, 'checkpoint or Actor LR differs')
    old = read(OLD)
    source = read(Path(old['python_output'])/'evaluation_complete.json')
    records = source['per_episode']
    require(len(records)==100,'historical completed reference differs')
    with tempfile.TemporaryDirectory(prefix='bg128-discovery-preflight-') as folder:
        temp=Path(folder)
        check_java(cfg,checked(suite['java_helper']),temp)
        control=temp/'control'
        trial_suite=dict(suite,control_directory=str(control))
        cfg_path=checked(suite['phase_config'])
        first=cfg['jobs'][0]
        with patch.object(discovery_budget,'LEDGER',temp/'test_ledger.json'), patch.object(discovery_budget,'previous_budget',return_value=prior):
            ledger=discovery_budget.DiscoveryLedger(trial_suite)
            publish=_publisher(trial_suite,ledger)
            MaterialPhaseCollector(cfg,cfg_path,publish_ready=publish,startup=dict(
                control_directory=str(control),campaign_id=cfg['campaign_id'],master_plan_sha256=suite['master_plan']['sha256']))
            try:
                publish(first,cfg_path)
            except (RuntimeError,ValueError):
                pass
            else:
                raise AssertionError('PLANNED slot allowed ready')
            ledger.claim(first)
            publish(first,cfg_path)
            stored=read(discovery_budget.LEDGER)
            require(stored['slots'][str(FIRST_RUN)]['status']=='CLAIMED' and
                    ('config_sha256='+suite['phase_config']['sha256']) in (control/'ready'/f'run_{FIRST_RUN}.properties').read_text(),
                    'durable claim or ready config bytes differ')
            # Verify the receipt transition contract using an old admitted
            # manifest as a temporary fixture. Real collection is unmodified.
            dataset_ref=source['source_refs'][0]
            physical_receipt=dict(run_index=FIRST_RUN,experiment_seed=first['experiment_seed'],
                phase_config=suite['phase_config'],normal_completion=True,
                references=dict(episode_manifest=dataset_ref))
            ledger.complete(first,dataset_ref,physical_receipt)
            require(ledger.resume()['jobs'][str(FIRST_RUN)]['status']=='COMPLETE', 'durable completion transaction failed')
            second=cfg['jobs'][1]
            ledger.claim(second)
            ledger.unknown(second,'controlled test failure')
            require(ledger.snapshot()['jobs'][str(FIRST_RUN+1)]['status']=='UNKNOWN','durable UNKNOWN transition failed')
        for j, r in zip(suite['jobs'],records):
            for k in ('run_index','actor_sampling_seed'):
                r['identity'][k]=j[k]
            r['identity']['environment_seed']=j['experiment_seed']
        fixture=temp/'fixture.json'
        write(fixture,source)
        with patch.object(run_discovery,'REPORT',temp/'report'):
            run_discovery.discovery_catalog(suite,dict(reports=dict(evaluation=reference(fixture))))
            inventory=read(temp/'report/discovery_result.json')
            require(inventory['first_window_failures']==2 and len(inventory['all_shifts'])==COUNT,
                    'discovery catalog lost observed failures or completed shifts')
    require(not OUTPUT.exists(),'offline preflight created runtime output')
    write(REPORT/'preparation_validation.json',dict(status='PASS_OFFLINE_PREPARATION',suite=reference(SUITE),
        all_java_identity_rows=100,new_environment_and_actor_seed_rows=100,java_initial_states=500,
        prior_merged_shifts=21699,original_model_digest=model.policy_version,protected_core=reference(CORE),
        test_checks=['original checkpoint load','collector admission config','planned slot blocks ready',
            'durable claim + Java ready publication','durable completion and UNKNOWN','100-row complete catalog fixture'],
        physical_shifts_started=0,optimizer_updates=0))
    print('PASS complete offline preparation')


if __name__=='__main__':
    main()
