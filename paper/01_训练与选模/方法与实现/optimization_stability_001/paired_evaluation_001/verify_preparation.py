"""Read-only final-u13 binding, paired seeds and Java100 accessors; no launch."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from business_graph_stability_evaluation_v1.gate import PLAN,REPORT,preflight,reference,check_sources,PROTECTED_SHA
    from business_graph_stability_evaluation_v1.budget import GLOBAL_PATH,preceding_reference,budget_snapshot
    from business_graph_stability_evaluation_v1.runtime import _load_model,_portable_state_fingerprint
    from business_graph_stability_campaign_v1.plan_io import write
    from business_graph_stability_campaign_v1.pairing import evaluation_source
    before=preceding_reference();own=reference(GLOBAL_PATH) if GLOBAL_PATH.exists() else None
    path=PLAN/'prepared_001/suite.json';suite,cfg,approval=preflight(path)
    assert approval is not None
    assert all(new[k]==old[k] for new,old in zip(suite['jobs'],evaluation_source()['jobs'])
               for k in ('actor_seed','experiment_seed','actor_sampling_seed'))
    torch.set_num_threads(2)
    rng=torch.get_rng_state().clone();model=_load_model(suite)
    assert model.update_index==13 and torch.equal(rng,torch.get_rng_state())
    fingerprint=_portable_state_fingerprint(model)
    assert fingerprint==_portable_state_fingerprint(model)
    probe='''public class Probe {
      public static void main(String[] args) {
        if(Baseline1CampaignPlan.jobCount()!=100 || Baseline1CampaignPlan.frozenPlanJobCount()!=100 || Baseline1CampaignPlan.resumeOffset()!=0) throw new AssertionError("count");
        for(int i=0;i<100;i++) {
          if(Baseline1CampaignPlan.dispatchPolicy(i)!=3 || Baseline1CampaignPlan.offlineEnabled(i) || Baseline1CampaignPlan.eventRecordingEnabled(i) || !Baseline1CampaignPlan.materialEnabled(i) || Baseline1CampaignPlan.waitWeight(i)!=.2) throw new AssertionError("mode");
          System.out.println(i+"\\t"+Baseline1CampaignPlan.runIndex(i)+"\\t"+Baseline1CampaignPlan.environmentSeed(i)+"\\t"+Baseline1CampaignPlan.actorSamplingSeed(i)+"\\t"+Baseline1CampaignPlan.recordingId(i)+"\\t"+Baseline1CampaignPlan.materialRoot(i));
        }
      }
    }'''
    with tempfile.TemporaryDirectory(prefix='bg128-paired-eval100-java-') as directory:
        p=Path(directory);(p/'Probe.java').write_text(probe)
        subprocess.run(['javac','-d',str(p),suite['java_helper']['path'],str(p/'Probe.java')],capture_output=True,text=True,check=True,timeout=60)
        observed=subprocess.run(['java','-cp',str(p),'Probe'],capture_output=True,text=True,check=True,timeout=60)
    lines=[row.split('\t') for row in observed.stdout.splitlines()];assert len(lines)==100
    for i,(values,j) in enumerate(zip(lines,suite['jobs'])):
        assert values==[str(i),str(j['run_index']),str(j['experiment_seed']),str(j['actor_sampling_seed']),j['recording_id'],j['material_event_directory']]
    assert preceding_reference()==before and own==(reference(GLOBAL_PATH) if GLOBAL_PATH.exists() else None)
    assert not Path(suite['python_output']).exists() and not Path(suite['python_output']+'_materials').exists()
    assert reference(ROOT.parents[1]/'python/episode_budget_ppo.py')['sha256']==PROTECTED_SHA
    check_sources(suite['source_pins'])
    snapshot=budget_snapshot()
    result=dict(status='PASS_PAIRED_EVAL100_PREPARED_NOT_STARTED',suite=reference(path),
        config=suite['phase_config'],checkpoint=suite['parent_artifacts']['checkpoint'],
        checks=dict(source_and_configuration_bytes=True,preflight=True,exact100_source_seed_pairs=True,
            fresh_run_recording_and_directories=True,sealed_new_u13_loads=True,load_keeps_global_rng=True,
            java_compiled=True,all100_java_identity_rows=True,all_ledgers_readonly=True,protected_core_unchanged=True),
        model_state_fingerprint=fingerprint,merged_reserved_before=snapshot['maximum_total_shifts'],
        merged_reserved_if_registered=suite['maximum_total_if_approved'],physical_shifts_prepared=100,
        new_physical_shifts=0,optimizer_steps=0,listener_started=False,anylogic_started=False,budget_registered=False)
    write(REPORT/'preparation_verification.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
