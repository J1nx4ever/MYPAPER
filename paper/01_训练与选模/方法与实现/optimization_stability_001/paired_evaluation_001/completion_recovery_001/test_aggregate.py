import importlib.util
from pathlib import Path
import numpy as np
import torch

from business_graph_summary_v1.test_learner import make_model,synthetic_batch,assert_tree_equal
from business_graph_summary_v1.evaluation import evaluate_episodes
from single_critic_task_query_v1.evaluation import _queue_samples
from single_critic_task_query_v1.policy import replay_policy

p=Path(__file__).with_name('aggregate_records.py')
spec=importlib.util.spec_from_file_location('stream_aggregate',p)
aggregate_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(aggregate_module)


def test_aggregate_matches_official_multi_episode_including_empty_and_failed_windows():
    count=torch.get_num_threads();torch.set_num_threads(1)
    try:
        model=make_model();episodes=synthetic_batch(model)
        full=evaluate_episodes(episodes,model,expected_complete_shifts=2,checkpoint_provenance={'test':'same'})
        records=[];queues=[];entropies=[]
        for episode in episodes:
            report=evaluate_episodes([episode],model,expected_complete_shifts=1,checkpoint_provenance={'test':'same'})
            records.extend(report['per_episode'])
            queues.append(_queue_samples(episode['states'][:-1]))
            with torch.no_grad():
                entropies.append(replay_policy(model.actor,episode['states'][:-1])['entropies'].cpu().numpy().astype(float))
        recovered=aggregate_module.aggregate(records,queues,entropies,model,{'test':'same'})
        # BG wrapper adds only representation metadata after official aggregation.
        assert_tree_equal({k:v for k,v in full.items() if k!='representation'},recovered)
        assert recovered['headline_metrics']['empty_windows']==4
        assert recovered['headline_metrics']['complete_task_rate']==0
    finally:torch.set_num_threads(count)
