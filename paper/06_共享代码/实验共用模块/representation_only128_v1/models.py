"""128D-only learned observations with the original parameterized interface.

Raw 1519D observations remain available to compute and certify the physical
summary and apply the external legal-action mask. Neither learned state encoder
receives their values: both receive constant zeros. Actor and Critic consume all
128 summary coordinates at the current decision and every causal history token.
Keeping the unused raw branch preserves parameter counts and initialization.
"""
import torch
from torch.distributions import Categorical
from grade_commitment_ppo_v1.models import (
    Actor as ReferenceActor, CommitmentCostCritic as ReferenceCritic,
    _states, _current_summary, _stamp, normalize_feature_config,
    feature_config_sha256, digest)
from .features_cache import (EXECUTION_VERSION, SummarySequence, SummaryFeatureCache,
    pack_summary_prefixes, validate_sequence, require)
from .contract import CANDIDATES, mask, ablation_contract, parameter_digest

def _history(encoder, gru, prefix, feature_mask):
    values = torch.tensor(prefix.features.copy(), dtype=torch.float32).unsqueeze(0)
    outputs, _ = gru(encoder(values * feature_mask))
    return outputs[0, list(prefix.query_indices)]

def _buffers(module, candidate):
    module.register_buffer('_summary_mask', torch.tensor(mask(candidate)))
    module.register_buffer('_ablation_contract', _stamp(ablation_contract(candidate)))
    module.register_buffer('_execution_contract', _stamp(EXECUTION_VERSION))

class Actor(ReferenceActor):
    def __init__(self, spec, *, candidate_id='e01_R-128', feature_config=None, gru_hidden=None):
        # Initialize precisely the same modules, in the same order, in EVERY arm.
        super().__init__(spec, candidate_id='a1_actor_gru', feature_config=feature_config, gru_hidden=gru_hidden)
        self.candidate_id = candidate_id
        _buffers(self, candidate_id)

    def forward(self, states, prefixes=None):
        _states(states)
        legal = states[:, 19:23]
        require(bool(((legal == 0) | (legal == 1)).all()) and bool(legal.bool().any(dim=1).all())
                and bool((states[:, 17] == 0).all()), 'invalid action mask or terminal request')
        current = validate_sequence(prefixes, states, self.feature_config)
        logits = self.base_head(torch.cat((self.state_encoder(torch.zeros_like(states)),
                    self.summary_encoder(current * self._summary_mask)), dim=-1))
        logits = logits + self.history_projection(_history(self.summary_encoder, self.history_gru,
                                                           prefixes, self._summary_mask))
        require(bool(torch.isfinite(logits).all()), 'nonfinite ablation logits')
        return logits.masked_fill(~legal.bool(), -torch.inf)

    def distribution(self, states, prefixes=None):
        return Categorical(logits=self(states, prefixes))

class CommitmentCostCritic(ReferenceCritic):
    def __init__(self, spec, *, feature_config=None, candidate_id='e01_R-128'):
        super().__init__(spec, feature_config=feature_config)
        _buffers(self, candidate_id)

    def forward(self, states, prefixes=None):
        _states(states)
        current = validate_sequence(prefixes, states, self.feature_config)
        return self.fusion(torch.cat((self.state_encoder(torch.zeros_like(states)),
            self.summary_encoder(current * self._summary_mask),
            _history(self.summary_encoder, self.history_gru, prefixes, self._summary_mask)), dim=-1)).squeeze(-1)

def actor_from_config(config):
    return Actor(config['network'], candidate_id=config['candidate_id'], feature_config=config['feature_config'])

def common_actor_digest(actor):
    return parameter_digest(actor)

MaterialCostCritic = CommitmentCostCritic
SummaryPrefixes = SummarySequence
