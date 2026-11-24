"""Freeze the explicitly requested seed pairs; no run, ledger or listener."""
import csv
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))


def main():
    from business_graph_stability_campaign_v1.pairing import contract,training_source,evaluation_source,TARGET_TRAIN,TARGET_EVAL
    from business_graph_stability_campaign_v1.plan_io import PLAN,write,reference
    source_contract=contract()
    rows=[]
    for role,source,campaign,run_base in [('train',training_source(),TARGET_TRAIN,9960001),
                                        ('inference',evaluation_source(),TARGET_EVAL,9970001)]:
        for ordinal,old in enumerate(source['jobs']):
            phase=old['phase_id']
            record=campaign+'-'+phase
            rows.append(dict(role=role,ordinal=ordinal,phase=phase,
                source_run_index=old['run_index'],source_recording_id=old['recording_id'],
                run_index=run_base+ordinal,recording_id=record,
                actor_seed=old['actor_seed'],experiment_seed=old['experiment_seed'],
                actor_sampling_seed=old['actor_sampling_seed']))
    write(PLAN/'pairing_manifest.json',dict(schema='bg128-explicit-612-seed-pairs-v1',
        pairing=source_contract,total_new_physical_shifts=612,training_shifts=512,inference_shifts=100,rows=rows))
    with (PLAN/'paired_seed_mapping.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    directory=PLAN/'paired_evaluation_001'
    future=ROOT/'runs'/TARGET_TRAIN/'arms/s01_BG128/updates/0013/checkpoint.pt'
    write(directory/'seed_plan.json',dict(schema='bg128-paired-evaluation-awaiting-checkpoint-v1',
        status='AWAITING_NEW_U13_CHECKPOINT',campaign_id=TARGET_EVAL,checkpoint_update=13,
        expected_checkpoint_path=str(future),checkpoint_sha256=None,final_config_sha256=None,
        seed_manifest=reference(PLAN/'pairing_manifest.json'),
        recording_id=TARGET_EVAL+'-s01_BG128_eval0100',
        python_output=str(ROOT/'runs'/TARGET_EVAL/'evaluation'),
        anylogic_event_directory=str(ROOT/'runs'/(TARGET_EVAL+'_materials')/'s01_BG128_eval0100'),
        jobs=[row for row in rows if row['role']=='inference'],
        optimizer_steps=0,physical_shifts=100,budget_registered=False))
    print(reference(PLAN/'pairing_manifest.json'))


if __name__=='__main__':main()
