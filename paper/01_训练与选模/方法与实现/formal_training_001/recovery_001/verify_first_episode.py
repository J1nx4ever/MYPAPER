"""Recover already completed run 9900001 offline, without another physical shift.

The original raw/config/Actor/journal bytes and run identity are immutable.
Only the BG-specific derived episode and its strong-validation evidence are new.
There is deliberately no optimizer, budget mutation, RPC or simulator call.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from business_graph_summary_recovery_v1.admission import admit_episode, reference
from business_graph_summary_recovery_v1.collection import (
    RECEIPT_SCHEMA, TRAINING_REWARD, _references, load_admitted_episode,
)
from business_graph_summary_recovery_v1.plan_io import load_suite
from business_graph_summary_v1.learner import JointCostPPO
from business_graph_summary_v1.models import digest
from business_graph_summary_v1.targets import verify_episode_targets
from business_graph_summary_storage_v1.episode import read_episode_hdf5
from grade_event_memory_v1.journal import read_journal

PLAN=Path(__file__).resolve().parent
OLD=PLAN.parent/'prepared_001'
REPORT=ROOT/'reports/token_attention_001/business_graph_summary_009/formal_training_001/recovery_001'


def immutable_json(path,value):
    data=(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:
            raise ValueError('immutable recovery evidence differs: '+str(path))
    else:
        with path.open('xb') as stream:stream.write(data)
    return reference(path)


def main():
    torch.set_num_threads(2)
    old_suite_path=OLD/'suite.json'
    suite,phases=load_suite(old_suite_path)
    phase=phases[0];phase_ref=suite['phase_configs'][0];job=phase['jobs'][0]
    assert job['run_index']==9900001 and job['experiment_seed']==255314794623997
    output=Path(phase['python_output']);raw=output/'episodes'/'run_9900001'
    actor=output/'actor.pt';journal=Path(job['material_event_bundle'])/'material_events.bin'
    destination=Path(suite['python_output'])/'recovery_001'/'derived'/'run_9900001.h5'
    originals=[old_suite_path,OLD/'approval.json',Path(phase_ref['path']),actor,journal,
               raw/'receipt.json',raw/'manifest.json',raw/'samples.h5',raw/'episodes.json']
    before={str(p):reference(p)['sha256'] for p in originals}
    raw_receipt=json.loads((raw/'receipt.json').read_bytes())
    assert raw_receipt['normal_completion'] is True and raw_receipt['transitions']==184 and raw_receipt['four_windows']==4
    if destination.exists():
        saved=json.loads((REPORT/'admission_report.json').read_bytes())
        assert saved['output']==reference(destination)
        admission=saved
    else:
        admission=admit_episode(raw,journal,config_path=phase_ref['path'],actor_path=actor,
            output_path=destination,split=job['partition'])
    admission_ref=immutable_json(REPORT/'admission_report.json',admission)
    journal_data=read_journal(journal)
    receipt=dict(schema=RECEIPT_SCHEMA,normal_completion=True,identity=admission['identity'],
        phase_id=phase['phase_id'],role=job['role'],raw_receipt=raw_receipt,
        run_index=job['run_index'],experiment_seed=job['experiment_seed'],phase_config=phase_ref,
        references=_references(phase_ref['path'],actor,journal,raw/'manifest.json',destination),
        four_windows=4,transitions=admission['transitions'],training_objective=TRAINING_REWARD,
        composite_contract=phase['composite_contract'],material_evidence=True,
        production_guarantee='pending',production_failures_retained=True)
    receipt_ref=immutable_json(REPORT/'material_receipt.json',receipt)
    # This rereads and rebinds raw RPC, phase, frozen Actor, causal event tape,
    # private sampling RNG and all MC costs; a bare storage callback is not enough.
    episode=load_admitted_episode(receipt)
    storage_episode=read_episode_hdf5(destination,expected_identity=episode['identity'],
        validator=lambda derived: verify_episode_targets(derived))
    initial=json.loads((Path(suite['arms'][0]['python_output'])/'initial_checkpoint.json').read_bytes())
    model=JointCostPPO.from_checkpoint(initial['path'],expected_config=suite['arms'][0]['learner_config'],
                                     expected_sha256=initial['sha256'],restore_global_rng=False)
    model.bind_actor_snapshot(actor)
    actor_before=model.policy_version;critic_before=digest(model.critic)
    values=model.predict_values(episode)
    assert values.shape==(185,) and values[-1]==0 and np.isfinite(values).all()
    target=np.asarray(episode['targets'].mc_returns)
    variance=float(np.var(target))
    physical=dict(ore_tons=episode['evaluation']['ore_tons'],waste_tons=episode['evaluation']['waste_tons'],
                  total_wait_seconds=episode['evaluation']['total_wait_seconds'])
    metrics=dict(critic_mc_mse=float(np.mean((values[:-1]-target)**2)),
        explained_variance=None if variance<=1e-12 else float(1-np.var(target-values[:-1])/variance),
        combined_cost=float(sum(episode['targets'].step_costs)),physical=physical,
        interpretation='fixed initial checkpoint inference only; no fit or policy-quality claim')
    for p in originals:assert reference(p)['sha256']==before[str(p)]
    checks=dict(original_bytes_unchanged=True,original_slot_run_seed_recording_preserved=True,
        raw_normal_completion_184_transitions_four_windows=True,real_anylogic_rpc_provenance=True,
        raw_actor_material_rpc_causal_alignment=True,private_sampling_rng_and_old_log_probabilities_replayed=True,
        bg_only_graph_identity_and_same_cost_certificate=True,canonical_hdf5_write_strong_read_closed=True,
        collection_load_admitted_episode_closed=True,MC_costs_rebuilt_from_actual_events=True,
        original_initial_checkpoint_and_actor_same_policy=model.policy_version==episode['identity']['policy_version'],
        finite_185_values_terminal_exact_zero=True,actor_critic_parameters_unchanged=(model.policy_version==actor_before and digest(model.critic)==critic_before),
        zero_optimizer_updates=(model.update_index==0 and not model.actor_optimizer.state and not model.critic_optimizer.state),
        storage_read_returns_same_original_states=np.array_equal(storage_episode['states'],episode['states']))
    assert all(checks.values())
    source_files=[ROOT/'business_graph_summary_storage_v1'/'episode.py',
        ROOT/'business_graph_summary_storage_v1'/'__init__.py']
    source_files += [ROOT/'business_graph_summary_recovery_v1'/name for name in
                     ('admission.py','collection.py','storage.py','targets.py','models.py','learner.py','policy.py','behavior.py','c_features.py','features_cache.py','plan_io.py','contract.py')]
    evidence=dict(schema='bg128-first-episode-reconciliation-evidence-v1',status='PASS',
        original_campaign=reference(old_suite_path),original_approval=reference(OLD/'approval.json'),
        original_job=job,raw_receipt=reference(raw/'receipt.json'),raw_manifest=reference(raw/'manifest.json'),
        raw_samples=reference(raw/'samples.h5'),raw_episodes=reference(raw/'episodes.json'),
        material_journal=reference(journal),actor=reference(actor),config=phase_ref,
        material_receipt=receipt_ref,admission_report=admission_ref,
        derived_h5=reference(destination),derived_manifest=reference(str(destination)+'.manifest.json'),
        derived_seal=reference(str(destination)+'.seal.json'),initial_checkpoint=initial,
        source_pins={str(p):reference(p)['sha256'] for p in source_files},checks=checks,
        transitions=184,states=185,windows=4,run_index=9900001,environment_seed=255314794623997,
        raw_process_calls=journal_data.raw_process_calls,metrics=metrics,
        optimizer_updates=0,rpc_started=False,anylogic_started=False,budget_mutated=False,
        complete_32_episode_batch_claim=False,additional_physical_shifts=0)
    ref=immutable_json(REPORT/'first_episode_evidence.json',evidence)
    print(json.dumps(dict(status='PASS',evidence=ref,transitions=184,optimizer_updates=0),ensure_ascii=False))


if __name__=='__main__':main()
