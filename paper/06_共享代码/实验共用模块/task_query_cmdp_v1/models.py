"""Actual S1519 five-branch network, with recurrence only in Grade Critic.

Supported execution: CPU float32, autocast disabled (the reused entity encoder
has this explicit domain).  The episode interface unrolls the entire causal
Grade Context sequence without detach, burn-in, padding, or state reuse.
"""
from __future__ import annotations

import hashlib

import torch
from torch import nn

from relation_gat_ego_v1.token_encoder import TokenEncoder
from .features import (SceneConfig, task_features, relation_features,
                       require_states, require_failed_before)


def mlp(*dimensions):
    layers = []
    for index, (left, right) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(nn.Linear(left, right))
        if index != len(dimensions) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class CrossAttention(nn.Module):
    """Global 32-entity visibility with bounded learned relation soft bias."""

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(128, 128)
        self.k = nn.Linear(64, 128)
        self.v = nn.Linear(64, 128)
        self.out = nn.Linear(128, 128)
        self.relation = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 4), nn.Tanh())
        self.norm = nn.LayerNorm(128)

    def forward(self, query, entities, relations):
        batch, queries, _ = query.shape
        q = self.q(query).reshape(batch, queries, 4, 32).transpose(1, 2)
        k = self.k(entities).reshape(batch, 32, 4, 32).transpose(1, 2)
        v = self.v(entities).reshape(batch, 32, 4, 32).transpose(1, 2)
        bias = self.relation(relations).permute(0, 3, 1, 2)
        weights = ((q @ k.transpose(-1, -2)) / (32 ** .5) + bias).softmax(-1)
        context = (weights @ v).transpose(1, 2).reshape(batch, queries, 128)
        return self.norm(query + self.out(context))


class ActorBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_encoder = mlp(130, 128, 128)
        self.action_id = nn.Parameter(torch.empty(4, 128))
        nn.init.normal_(self.action_id, std=.02)
        self.attention = CrossAttention()
        self.head = mlp(192, 128, 1)

    def forward(self, entities, states, fifo, task, relation):
        trucks = states[:, 27:1107].reshape(-1, 9, 120)
        ego_id = trucks[:, :, 0].argmax(-1)
        ego = entities[torch.arange(len(states)), ego_id]
        query = torch.cat((ego[:, None, :].expand(-1, 4, -1), entities[:, 9:13],
                           states[:, 19:23, None], states[:, 23:27, None]), -1)
        query = self.query_encoder(query) + self.action_id
        context = self.attention(query, entities, relation)
        common = torch.cat((fifo, task), -1)[:, None, :].expand(-1, 4, -1)
        raw = self.head(torch.cat((context, common), -1)).squeeze(-1)
        masked = raw.masked_fill(states[:, 19:23] == 0, float('-inf'))
        return masked, context


class ValueBranch(nn.Module):
    def __init__(self, *, recurrent=False):
        super().__init__()
        self.query = nn.Parameter(torch.empty(128))
        nn.init.normal_(self.query, std=.02)
        self.attention = CrossAttention()
        if recurrent:
            self.gru = nn.GRU(128, 64, num_layers=1, dropout=0, bidirectional=False)
        self.head = mlp(256 if recurrent else 192, 128, 1)

    def context(self, entities, relations):
        query = self.query[None, None, :].expand(len(entities), 1, -1)
        return self.attention(query, entities, relations).squeeze(1)


class FiveBranchNetwork(nn.Module):
    """One shared entity/FIFO/Task base and five independent task readouts."""

    def __init__(self):
        super().__init__()
        self.entity_encoder = TokenEncoder()
        self.fifo_encoder = nn.Sequential(mlp(48, 64, 32), nn.ReLU())
        self.task_encoder = nn.Sequential(mlp(15, 32, 32), nn.ReLU())
        self.actor = ActorBranch()
        self.wait_critic = ValueBranch()
        self.grade_critic = ValueBranch(recurrent=True)
        self.ore_critic = ValueBranch()
        self.waste_critic = ValueBranch()

    def _common(self, states, grade_failed_before, config):
        require_states(states, config)
        failed = require_failed_before(grade_failed_before, states)
        # Validate the module domain too, rather than fail later with a mixed
        # device or silently misrepresent inherited CPU-only validation support.
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError('FiveBranchNetwork supports CPU float32 parameters only')
        entities = self.entity_encoder(states)
        raw_fifo = states[:, 1391:1439]
        present = raw_fifo.reshape(-1, 16, 3)[:, :, 0].bool().any(-1)
        fifo = self.fifo_encoder(raw_fifo) * present[:, None].to(states.dtype)
        task = self.task_encoder(task_features(states, failed, config))
        actor_rel = relation_features(states, config, actor=True)
        critic_rel = relation_features(states, config, actor=False)
        return entities, fifo, task, failed, actor_rel, critic_rel

    def _forward(self, states, grade_failed_before, config, hidden, *, episode):
        entities, fifo, task, failed, actor_rel, critic_rel = self._common(
            states, grade_failed_before, config)
        logits, actor_context = self.actor(entities, states, fifo, task, actor_rel)
        contexts = {name: getattr(self, name + '_critic').context(entities, critic_rel)
                    for name in ('wait', 'grade', 'ore', 'waste')}
        batch_size = 1 if episode else len(states)
        if hidden is None:
            hidden = states.new_zeros((1, batch_size, 64))
        if (hidden.shape != (1, batch_size, 64) or hidden.device != states.device or
                hidden.dtype != states.dtype or not bool(torch.isfinite(hidden).all())):
            raise ValueError('Grade hidden must be finite CPU float32 [1,B,64]')
        grade_input = contexts['grade'].unsqueeze(1 if episode else 0)
        recurrent, final_hidden = self.grade_critic.gru(grade_input, hidden)
        grade_memory = recurrent.squeeze(1 if episode else 0)
        common = torch.cat((fifo, task), -1)
        wait = self.wait_critic.head(torch.cat((contexts['wait'], common), -1)).squeeze(-1)
        grade_logits = self.grade_critic.head(torch.cat((
            contexts['grade'], grade_memory, common), -1)).squeeze(-1)
        ore_logits = self.ore_critic.head(torch.cat((contexts['ore'], common), -1)).squeeze(-1)
        waste_logits = self.waste_critic.head(torch.cat((contexts['waste'], common), -1)).squeeze(-1)
        return {
            'logits': logits, 'masked_logits': logits,
            'wait': wait, 'grade': (1 - failed) * grade_logits.sigmoid(),
            'ore': ore_logits.sigmoid(), 'waste': waste_logits.sigmoid(),
            'grade_logits': grade_logits, 'ore_logits': ore_logits, 'waste_logits': waste_logits,
            'final_hidden': final_hidden, 'actor_context': actor_context,
            'grade_context': contexts['grade'], 'grade_memory': grade_memory,
        }

    def forward_step(self, states, grade_failed_before, config: SceneConfig, hidden=None):
        """Independent parallel episodes, each advanced exactly one DECISION."""
        return self._forward(states, grade_failed_before, config, hidden, episode=False)

    def forward_episode(self, states, grade_failed_before, config: SceneConfig):
        """One complete unpadded episode, h0=0, full causal BPTT."""
        if len(states) > 512:
            raise ValueError('episode exceeds 512 DECISION resource contract; do not truncate')
        failed = torch.as_tensor(grade_failed_before)
        if failed.ndim == 1 and len(failed) > 1 and bool((failed[1:] < failed[:-1]).any()):
            raise ValueError('grade_failed_before cannot revert within an episode')
        return self._forward(states, grade_failed_before, config, None, episode=True)

    def forward(self, states, grade_failed_before, config: SceneConfig, hidden=None):
        return self.forward_step(states, grade_failed_before, config, hidden)

    def parameter_owners(self):
        """Unique named ownership; shared modules receive all five task losses."""
        shared = {'entity_encoder', 'fifo_encoder', 'task_encoder'}
        return {name: ('shared' if name.split('.')[0] in shared else name.split('.')[0])
                for name, _ in self.named_parameters()}

    def shared_parameters(self):
        for module in (self.entity_encoder, self.fifo_encoder, self.task_encoder):
            yield from module.parameters()

    def behavior_digest(self):
        digest = hashlib.sha256()
        for name, value in self.state_dict().items():
            if name.split('.')[0] not in {'entity_encoder', 'fifo_encoder', 'task_encoder', 'actor'}:
                continue
            digest.update(name.encode('utf-8') + b'\0')
            digest.update(str(tuple(value.shape)).encode('ascii') + b'\0')
            digest.update(str(value.dtype).encode('ascii') + b'\0')
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    actor_fingerprint = behavior_digest


__all__ = ['FiveBranchNetwork', 'CrossAttention']
