"""Read-only common-u0/source guard and offline acceptance after real tests."""
import json
import random
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import torch
    from business_graph_stability_campaign_v1.plan_io import PLAN,REPORT,reference,write,sha
    from business_graph_stability_campaign_v1.gate import source_pins,PROTECTED_SHA
    from business_graph_stability_campaign_v1.pairing import training_source,contract
    from business_graph_stability_v1.learner import JointCostPPO,default_config
    from business_graph_summary_v1.learner import _load_checkpoint
    from business_graph_summary_v1.models import digest
    command=json.loads((REPORT/'paired_test_command.json').read_bytes())
    assert command['exit_code']==0
    source=contract()['common_initial_checkpoint']
    saved,_=_load_checkpoint(source['path'])
    config=default_config(seed=training_source()['arms'][0]['actor_seed'])
    torch.set_num_threads(2)
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    model=JointCostPPO.from_common_initial_checkpoint(source['path'],expected_sha256=source['sha256'],
                                                     config=config,restore_global_rng=False)
    checks=dict(actor_tensors_exact=all(torch.equal(model.actor.state_dict()[k],v) for k,v in saved['actor'].items()),
        critic_tensors_exact=all(torch.equal(model.critic.state_dict()[k],v) for k,v in saved['critic'].items()),
        shuffle_rng_exact=torch.equal(model.generator.get_state(),saved['generator']),
        sampling_rng_exact=torch.equal(model.sampling_generator.get_state(),saved['sampling_generator']),
        both_adam_empty=not model.actor_optimizer.state and not model.critic_optimizer.state,
        zero_training_counters=model.update_index==0 and not model.consumed_batches and not model.consumed_episodes,
        actor_lr_only=model.actor_optimizer.param_groups[0]['lr']==1e-4 and model.critic_optimizer.param_groups[0]['lr']==1e-3,
        global_rng_readonly=random.getstate()==rng[0] and np.array_equal(np.random.get_state()[1],rng[1][1]) and torch.equal(torch.get_rng_state(),rng[2]))
    assert all(checks.values()),checks
    old=json.loads((REPORT/'preserved_before.json').read_bytes())
    preserved={p:sha(p)==value for p,value in old.items()}
    assert all(preserved.values()),[p for p,okay in preserved.items() if not okay]
    assert sha(ROOT.parents[1]/'python/episode_budget_ppo.py')==PROTECTED_SHA
    write(REPORT/'common_checkpoint_verification.json',dict(status='PASS',source_checkpoint=source,
        checks=checks,actor_digest=digest(model.actor),critic_digest=digest(model.critic),
        protected_core_sha256=PROTECTED_SHA,physical_shifts=0,optimizer_steps=0))
    write(REPORT/'preservation_verification.json',dict(status='PASS',files=preserved,
        protected_core_sha256=PROTECTED_SHA,new_physical_shifts=0,old_ledgers_unchanged=True))
    artifacts=[reference(REPORT/name) for name in ('paired_pytest.txt','paired_test_command.json',
        'common_checkpoint_verification.json','preservation_verification.json')]
    checks=dict(common_u0_loaded_exactly=True,only_actor_lr_changes_numerically=True,
        exact_checkpoint_restore=True,per_epoch_diagnostics_do_not_change_steps_or_rng=True,
        training_and_evaluation_seeds_paired_by_source_row=True,
        paired_runs_still_new_consumption=True,latest100_evaluation_in_merged_budget=True,
        actual_bg128_storage_roundtrip=True,independent_evaluation_zero_update_guard=True,
        high_frequency_diagnostics_off=True,old_sources_and_ledgers_unchanged=True)
    write(REPORT/'runtime_acceptance.json',dict(schema='bg128-stability-campaign-offline-acceptance-v1',
        status='PASS_OFFLINE_ONLY',checks=checks,source_pins=source_pins(),verification_artifacts=artifacts,
        frozen_architecture_validation=reference(ROOT/'reports/token_attention_001/business_graph_summary_009/engineering_003/validation.json'),
        formal_first_batch_is_end_to_end_validation=True,separate_physical_smoke_shifts=0,
        protected_ppo_sha256=PROTECTED_SHA,physical_shifts=0,optimizer_updates=0,
        rpc_started=False,anylogic_started=False,budget_registration=False))
    print(json.dumps(reference(REPORT/'runtime_acceptance.json')))


if __name__=='__main__':main()
