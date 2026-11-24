"""Declare the offline numerical reconstruction before it is executed."""
from common import *
from business_graph_summary_campaign_v1.plan_io import load_suite

def main():
    arm=PROJECT/'runs/business-graph-summary-formal-seed1-20261118-001/arms/s01_BG128'
    s,_=load_suite(PLAN.parent/'formal_training_001/prepared_001/suite.json')
    updates=[]
    for n in (14,15,16):
        summary=read(arm/f'updates/{n:04}/training_summary.json')
        ids=summary['transaction']['run_indices']
        jobs=[j for j in s['jobs'] if j['run_index'] in ids]
        updates.append(dict(update=n,before=ref(arm/f'updates/{n-1:04}/checkpoint.pt'),
            after=ref(arm/f'updates/{n:04}/checkpoint.pt'),summary=ref(arm/f'updates/{n:04}/training_summary.json'),
            actor=ref(arm/f'train_u{n:04}/actor.pt'),
            historical_runs=[dict(run_index=j['run_index'],environment_seed=j['experiment_seed'],
                                  actor_sampling_seed=j['actor_sampling_seed']) for j in jobs]))
    value=dict(schema='bg128-offline-minibatch-reconstruction-config-v1',
        diagnostic_id='bg128-late-minibatch-replay-20261118-001',cpu_threads=1,updates=updates,
        physical_shifts=0,new_recording_id=None,new_run_indices=[],new_seeds=[],
        anylogic_event_directory=None,python_output=str(OUT/'minibatch_replay_001'),
        source_training_suite=ref(PLAN.parent/'formal_training_001/prepared_001/suite.json'),
        no_official_checkpoint_writes=True,no_ledger_writes=True,
        numerical_method='Original unmodified learner.update on memory-only clones; detach-only observers.',
        acceptance='Exact final Actor/Critic/Adam/private shuffle RNG/target equality with original sealed checkpoint.',
        protected_core=ref(CORE),replay_implementation=ref(PLAN/'minibatch_replay.py'))
    p=PLAN/'minibatch_replay_config.json'
    with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(diagnostic_id=value['diagnostic_id'],config=ref(p),python_output=value['python_output'],
        anylogic_event_directory=None,new_recording_id=None,new_run_indices=[],new_seeds=[],
        historical_runs_replayed=sum(len(u['historical_runs']) for u in updates))))

if __name__=='__main__':main()
