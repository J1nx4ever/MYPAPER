"""Use the frozen mixed campaign with an explicitly sealed alternative network."""

from contextlib import contextmanager
from copy import deepcopy
from inspect import getsource, getsourcefile
from pathlib import Path
from textwrap import dedent
from types import FunctionType
from unittest.mock import patch

import torch

from business_graph_ablation_v1 import models
from business_graph_summary_campaign_v1 import admission as campaign_admission
from business_graph_summary_campaign_v1 import collection as campaign_collection
from business_graph_summary_campaign_v1.plan_io import reference
from business_graph_summary_v1 import admission as original_admission
from business_graph_summary_v1 import learner as original_learner
from business_graph_summary_v1.contract import require
from business_graph_summary_v1.models import digest
from grade_only_ppo_v2.runtime import _equal_tree


def source_files():
    return (
        Path(__file__).resolve(),
        *sorted(path for path in Path(models.__file__).resolve().parent.glob('*.py')
                if not path.name.startswith('test_')),
    )


def _with_globals(method, **overrides):
    rebound = FunctionType(method.__code__, dict(method.__globals__, **overrides),
                           method.__name__, method.__defaults__, method.__closure__)
    rebound.__kwdefaults__ = method.__kwdefaults__
    return rebound


def _actor_loader(actor_factory, signature):
    loader = original_admission.frozen_actor
    imports = ('    from .learner import (ACTOR_SCHEMA, validate_config, _validate_model_state,\n'
               '                          _load_checkpoint, _hash, implementation_signature)\n'
               '    from .models import actor_from_config, digest\n')
    source = dedent(getsource(loader))
    require(source.count(imports) == 1 and loader.__module__ == original_admission.__name__,
            'frozen Actor loader imports changed')
    scope = dict(loader.__globals__, ACTOR_SCHEMA=original_learner.ACTOR_SCHEMA,
                 validate_config=original_learner.validate_config,
                 _validate_model_state=original_learner._validate_model_state,
                 _load_checkpoint=original_learner._load_checkpoint, _hash=original_learner._hash,
                 implementation_signature=signature, actor_from_config=actor_factory, digest=digest)
    exec(compile(source.replace(imports, ''), getsourcefile(loader), 'exec'), scope)
    return scope[loader.__name__]


def _learner(runtime, suite):
    variant, contract = suite['ablation_variant'], suite['ablation']
    framework_config = suite['arms'][0]['learner_config']
    sources = {str(path): reference(path)['sha256'] for path in source_files()}
    require(all(suite['source_pins'].get(path) == sha for path, sha in sources.items()),
            'ablation implementation differs from the frozen suite')
    require(contract == models.architecture_contract(variant)
            and framework_config['feature_config'] == models.feature_config_for(variant),
            'ablation architecture or certified input contract differs')

    def signature():
        require(all(reference(path)['sha256'] == sha for path, sha in sources.items()),
                'ablation implementation changed during execution')
        result = original_learner.implementation_signature()
        result.update(ablation_variant=variant, ablation_contract=contract, ablation_sources_sha256=sources)
        return result

    def model_config(config):
        return dict(config, ablation_variant=variant, ablation_contract=contract,
                    network=models.ACTOR_SPECS[variant], memory_critic=models.CRITIC_SPECS[variant])

    def actor_factory(config):
        require(config == framework_config, 'ablation Actor configuration differs from its suite')
        return models.actor_from_config(model_config(config))

    def critic_factory(spec, *, feature_config, candidate_id, pair_seed):
        require(spec == framework_config['memory_critic'], 'unexpected original Critic contract')
        return models.critic_from_config(model_config(dict(feature_config=feature_config,
            candidate_id=candidate_id, seed=pair_seed)))

    extension = runtime._learner_class()
    initialize = _with_globals(extension.__init__, actor_from_config=actor_factory,
                               MaterialCostCritic=critic_factory)

    class AblationLearner(extension):
        def __init__(self, config):
            initialize(self, config)
            require(self.config == framework_config, 'ablation learner configuration differs from its suite')
            self.runtime_state['ablation_variant'] = variant
            self.runtime_state['ablation'] = deepcopy(contract)

        save_actor = _with_globals(extension.save_actor, implementation_signature=signature)
        save_checkpoint = _with_globals(extension.save_checkpoint, implementation_signature=signature)
        _restore = classmethod(_with_globals(extension.from_checkpoint.__func__, implementation_signature=signature))

        @classmethod
        def from_checkpoint(cls, *args, **kwargs):
            model = cls._restore(*args, **kwargs)
            require(model.runtime_state.get('ablation_variant') == variant
                    and model.runtime_state.get('ablation') == contract,
                    'checkpoint ablation identity differs')
            return model

    return AblationLearner, _actor_loader(actor_factory, signature)


def _compare_initializations(suite, model, checkpoint):
    recreated = type(model)(model.config)
    checks = {}
    storage_pointers = []
    for role in ('actor', 'critic'):
        current = getattr(model, role)
        independent = getattr(recreated, role)
        checks[f'{role}_tensor_bytes'] = _equal_tree(current.state_dict(), independent.state_dict())
        addresses = {p.untyped_storage().data_ptr() for p in current.parameters()}
        checks[f'{role}_storage_independent'] = addresses.isdisjoint(
            p.untyped_storage().data_ptr() for p in independent.parameters())
        storage_pointers.append(addresses)
    checks.update(actor_critic_storage_disjoint=storage_pointers[0].isdisjoint(storage_pointers[1]),
                  fresh_u0=all(item.update_index == 0 and not item.consumed_batches and not item.consumed_episodes
                               for item in (model, recreated)),
                  empty_optimizers=all(not item.actor_optimizer.state and not item.critic_optimizer.state
                                       for item in (model, recreated)),
                  minibatch_rng=torch.equal(model.generator.get_state(), recreated.generator.get_state()),
                  sampling_rng=torch.equal(model.sampling_generator.get_state(), recreated.sampling_generator.get_state()))
    require(all(checks.values()), 'ablation fresh initialization is not reproducible')
    from business_graph_summary_campaign_v1.runtime import _same_or_save

    output = Path(suite['arms'][0]['python_output']) / 'initialization_comparison.json'
    _same_or_save(output, dict(initial=checkpoint, checks=checks, architecture=suite['ablation'],
                              actor_seed=model.config['seed'], **model.initialization_fingerprints()))
    return reference(output)


@contextmanager
def runtime_scope(suite):
    """Install network and artifact adapters inside one serial controller process."""
    import mixed_runtime as runtime

    learner, load_actor = _learner(runtime, suite)
    variant, design = suite['ablation_variant'], suite['ablation']
    learning_contract = runtime._learning_contract
    evaluate_episodes = runtime.evaluate_episodes
    build_report = runtime.build_report

    def learning(settings, candidate):
        return dict(learning_contract(settings, candidate), ablation_variant=variant, ablation=design)

    def evaluation(*args, **kwargs):
        result = evaluate_episodes(*args, **kwargs)
        result['representation'].update(ablation_variant=variant, ablation=design)
        return result

    def evaluation_summary(*args, **kwargs):
        return dict(build_report(*args, **kwargs), ablation_variant=variant, ablation=design)

    with (
        runtime.VERIFY_LOCK,
        patch.multiple(runtime, _learner_class=lambda: learner,
                       _compare_initializations=_compare_initializations, _learning_contract=learning,
                       frozen_actor=load_actor, evaluate_episodes=evaluation, build_report=evaluation_summary),
        patch.object(original_admission, 'frozen_actor', load_actor),
        patch.object(campaign_admission, 'frozen_actor', load_actor),
        patch.object(campaign_collection, 'frozen_actor', load_actor),
    ):
        yield runtime
