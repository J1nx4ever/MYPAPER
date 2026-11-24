"""Independent current-state Actor and one causal material-memory cost Critic."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from python.constrained_ppo import Actor, digest
from grade_event_memory_v1.sequence import FEATURE_NAMES, PrefixBatch


class MaterialCostCritic(nn.Module):
    """One linear scalar expected future cost; no per-window heads or waiting."""

    def __init__(self, spec):
        super().__init__()
        layers, width = [], 1519
        for hidden in spec['state_hidden_sizes']:
            layers.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        self.state_encoder = nn.Sequential(*layers)
        self.event_encoder = nn.Sequential(nn.Linear(len(FEATURE_NAMES), spec['event_embedding']), nn.ReLU())
        self.gru = nn.GRU(spec['event_embedding'], spec['gru_hidden'], batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(width + spec['gru_hidden'], spec['fusion_hidden']),
                                    nn.ReLU(), nn.Linear(spec['fusion_hidden'], 1))

    def history(self, prefix: PrefixBatch):
        """Empty histories stay zero; padding never advances the recurrent state."""
        features = torch.tensor(np.asarray(prefix.features), dtype=torch.float32)
        lengths = torch.tensor(np.asarray(prefix.lengths), dtype=torch.int64)
        count = len(lengths)
        hidden = features.new_zeros((count, self.gru.hidden_size))
        chosen = torch.nonzero(lengths > 0, as_tuple=False).flatten()
        if chosen.numel():
            encoded = self.event_encoder(features[chosen])
            packed = pack_padded_sequence(encoded, lengths[chosen].cpu(), batch_first=True,
                                          enforce_sorted=False)
            _, final = self.gru(packed)
            hidden = hidden.index_copy(0, chosen, final[0])
        return hidden

    def forward(self, states, prefix: PrefixBatch):
        if states.ndim != 2 or states.shape[1] != 1519 or not torch.isfinite(states).all():
            raise ValueError('cost Critic requires finite states[N,1519]')
        if len(prefix.lengths) != len(states):
            raise ValueError('state and history cardinality differs')
        return self.fusion(torch.cat((self.state_encoder(states), self.history(prefix)), dim=-1)).squeeze(-1)


__all__ = ['Actor', 'digest', 'MaterialCostCritic']
