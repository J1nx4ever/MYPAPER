"""Independent physical and MC metrics with the unchanged TQ008 denominators."""
from single_critic_task_query_v1.evaluation import evaluate_episodes as _evaluate
from .contract import VERSION, ablation_contract, require


def evaluate_episodes(episodes, model=None, **kwargs):
    if model is not None:
        require(model.config.get('graph_contract') == ablation_contract(),
                'evaluated model is not the frozen business-graph architecture')
    result = _evaluate(episodes, model, **kwargs)
    result['representation'] = dict(version=VERSION, graph_contract=ablation_contract(),
        metric_implementation='single_critic_task_query_v1.evaluation.evaluate_episodes',
        metric_definition_changed=False, model_provided=model is not None)
    return result
