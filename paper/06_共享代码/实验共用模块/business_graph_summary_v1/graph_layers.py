"""Relation-conditioned messages normalized within each business group.

Road, commitment, service and material-flow records have separate incoming
softmax denominators. Edge attributes enter both score and value; a sole incoming
edge therefore still communicates its length, traffic and business payload.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .graph import (GraphBatch, NODE_NAMES, GROUP_NAMES, RELATION_NAMES,
                    EDGE_ATTRIBUTE_NAMES)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError('nonfinite business graph ' + name)
    return value


def checked_norm(layer, value, name):
    value = finite(value, name + ' input')
    centered = finite(value - finite(value.mean(-1, keepdim=True), name + ' mean'),
                      name + ' centered')
    finite(centered.square(), name + ' variance')
    return finite(layer(value), name)


class GraphLayer(nn.Module):
    """Four-head synchronous propagation; Global stays an unchanged bypass."""
    def __init__(self):
        super().__init__()
        edge_width = len(RELATION_NAMES) + len(EDGE_ATTRIBUTE_NAMES)
        self.score_src = nn.Linear(64, 64, bias=False)
        self.score_dst = nn.Linear(64, 64, bias=False)
        self.score_edge = nn.Linear(edge_width, 64, bias=False)
        self.score_a = nn.Parameter(torch.empty(4, 16))
        nn.init.uniform_(self.score_a, -.25, .25)
        self.value = nn.Linear(64, 64)
        self.edge_value = nn.Linear(edge_width, 64, bias=False)
        # No bias: absent groups contribute exactly zero before the residual.
        # Presence bits distinguish an absent group from a zero-valued message.
        self.output = nn.Linear(len(GROUP_NAMES) * 65, 64, bias=False)
        self.norm = nn.LayerNorm(64, eps=1e-5)

    def forward(self, h, graph, *, diagnostic=False):
        require(isinstance(h, torch.Tensor) and h.dtype == torch.float32 and
                h.device.type == 'cpu' and h.ndim == 3 and
                h.shape[1:] == (len(NODE_NAMES), 64) and len(h) > 0,
                'business graph requires CPU float32 H[B,19,64]')
        require(not torch.is_autocast_enabled('cpu'), 'graph autocast must be disabled')
        require(isinstance(graph, GraphBatch) and graph.batch_size == len(h),
                'matching business GraphBatch required')
        graph.validate()
        finite(h, 'input')
        flat = h.reshape(-1, 64)
        src, dst = graph.edge_index
        groups = len(GROUP_NAMES)
        edge_input = torch.cat((F.one_hot(graph.edge_type, len(RELATION_NAMES)).to(h.dtype),
                                graph.edge_attr), dim=-1)
        score_input = finite(self.score_src(flat)[src] + self.score_dst(flat)[dst] +
                             self.score_edge(edge_input), 'score input')
        scores = finite((F.leaky_relu(score_input, .2).reshape(-1, 4, 16) *
                         self.score_a).sum(-1), 'scores')
        message_input = finite(self.value(flat)[src] + self.edge_value(edge_input),
                               'message input')
        messages = finite(F.leaky_relu(message_input, .2).reshape(-1, 4, 16), 'messages')
        # Each (destination, semantic group) owns four independent head sums.
        buckets = dst * groups + graph.edge_group
        bucket_count = len(flat) * groups
        index = buckets[:, None].expand(-1, 4)
        maxima = scores.new_full((bucket_count, 4), -torch.inf).scatter_reduce(
            0, index, scores, reduce='amax', include_self=True)
        # Empty buckets intentionally have -inf; only present buckets enter exp.
        exponentials = finite((scores - maxima[buckets]).exp(), 'softmax exponentials')
        denominators = scores.new_zeros((bucket_count, 4)).index_add(0, buckets, exponentials)
        weights = finite(exponentials / denominators[buckets], 'group weights')
        weighted = finite(messages * weights[:, :, None], 'weighted messages')
        aggregate = flat.new_zeros((bucket_count, 4, 16)).index_add(0, buckets, weighted)
        aggregate = finite(aggregate.reshape(len(flat), groups, 64), 'group aggregate')
        counts = torch.bincount(buckets, minlength=bucket_count).reshape(len(flat), groups)
        presence = counts > 0
        fused_input = torch.cat((aggregate.flatten(1), presence.to(h.dtype)), dim=-1)
        projected = finite(self.output(fused_input), 'group fusion')
        pre_norm = finite(flat + projected, 'residual').reshape_as(h)
        # Global is represented by original global fields and read out directly;
        # it is not a synthetic communication shortcut or a graph destination.
        physical = checked_norm(self.norm, pre_norm[:, :18], 'physical norm')
        result = torch.cat((physical, h[:, 18:19]), dim=1)
        if diagnostic:
            return result, dict(edge_input=edge_input, scores=scores, weights=weights,
                                score_input=score_input, messages=messages,
                                message_input=message_input, buckets=buckets,
                                group_messages=aggregate.reshape(len(h), len(NODE_NAMES), groups, 64),
                                group_masks=presence.reshape(len(h), len(NODE_NAMES), groups),
                                group_counts=counts.reshape(len(h), len(NODE_NAMES), groups),
                                projected=projected.reshape_as(h), pre_norm=pre_norm)
        return result


__all__ = ['GraphLayer', 'finite', 'checked_norm']
