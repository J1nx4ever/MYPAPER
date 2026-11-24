"""CPU float32 synchronous relation propagation with incoming-record softmax."""
import torch
from torch import nn
from torch.nn import functional as F

from .contract import require
from .relation_edges import GraphEdges


class NeuralNumericRangeError(ValueError):
    """A finite input overflowed a required neural intermediate."""


def finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise NeuralNumericRangeError('nonfinite graph ' + label)
    return value


class GraphLayer(nn.Module):
    def __init__(self, mode='G2'):
        super().__init__()
        require(mode in ('G2', 'U2'), 'graph mode must be G2 or U2')
        self.mode = mode
        if mode == 'G2':
            self.score_dst = nn.Linear(64, 64, bias=False)
            self.score_src = nn.Linear(64, 64, bias=False)
            self.score_edge = nn.Linear(29, 64, bias=False)
            self.score_a = nn.Parameter(torch.empty(4, 16))
            nn.init.uniform_(self.score_a, -0.25, 0.25)
        self.value = nn.Linear(64, 64)
        self.edge_value = nn.Linear(29, 64, bias=False)
        self.output = nn.Linear(64, 64)
        self.count_projection = nn.Linear(27, 64, bias=False)
        self.norm = nn.LayerNorm(64, eps=1e-5)
        expected = 22144 if mode == 'G2' else 12032
        require(sum(p.numel() for p in self.parameters()) == expected, 'graph parameter inventory differs')

    def forward(self, h, edges, *, diagnostic=False):
        require(isinstance(h, torch.Tensor) and h.dtype == torch.float32 and h.device.type == 'cpu'
                and h.ndim == 3 and h.shape[1:] == (31, 64) and len(h) > 0,
                'graph H must be CPU float32[B,31,64]')
        require(not torch.is_autocast_enabled('cpu'), 'graph autocast must be disabled')
        require(isinstance(edges, GraphEdges), 'graph requires GraphEdges')
        edges.validate()
        require(edges.batch_size == len(h), 'graph batch sizes differ')
        finite(h, 'input')
        flat = h.reshape(-1, 64)
        src, dst = edges.edge_index
        edge29 = torch.cat((F.one_hot(edges.edge_type, 27).to(torch.float32), edges.edge_attr), dim=1)
        value = finite(self.value(flat), 'V')
        edge_value = finite(self.edge_value(edge29), 'edgeV')
        message_input = finite(value[src] + edge_value, 'message preactivation')
        messages = finite(F.leaky_relu(message_input, negative_slope=0.2).reshape(-1, 4, 16), 'message')
        scores = score_input = None
        if self.mode == 'G2':
            wd = finite(self.score_dst(flat), 'score Wdst')
            ws = finite(self.score_src(flat), 'score Wsrc')
            we = finite(self.score_edge(edge29), 'score We')
            score_input = finite(finite(wd[dst] + ws[src], 'score node sum') + we, 'score preactivation')
            activated = finite(F.leaky_relu(score_input, negative_slope=0.2).reshape(-1, 4, 16), 'score activation')
            products = finite(activated * self.score_a, 'score products')
            scores = finite(products.sum(dim=-1), 'scores')  # no /4: this is not Ego dot-product attention
            incoming_index = dst[:, None].expand(-1, 4)
            maxima = scores.new_full((len(flat), 4), -torch.inf).scatter_reduce(
                0, incoming_index, scores, reduce='amax', include_self=True)
            finite(maxima, 'incoming maxima')  # SELF guarantees every destination exists
            shifted = finite(scores - maxima[dst], 'shifted scores')
            exponentials = finite(shifted.exp(), 'softmax exponentials')
            denominators = finite(scores.new_zeros((len(flat), 4)).index_add(0, dst, exponentials),
                                  'softmax denominators')
            weights = finite(exponentials / denominators[dst], 'softmax weights')
        else:
            degree = edges.count.sum(dim=-1).reshape(-1).to(torch.float32)
            weights = (1 / degree[dst, None]).expand(-1, 4)
        weighted = finite(messages * weights[:, :, None], 'weighted messages')
        aggregate = finite(flat.new_zeros((len(flat), 4, 16)).index_add(0, dst, weighted), 'aggregate').reshape(-1, 64)
        projected = finite(self.output(aggregate), 'O')
        count_term = finite(self.count_projection(edges.degree_features.reshape(-1, 27)), 'count projection')
        pre_norm = finite(finite(flat + projected, 'residual plus O') + count_term, 'residual plus count')
        # Native LayerNorm can hide a float32 variance overflow behind a zero
        # reciprocal standard deviation. Reject that range explicitly instead
        # of silently returning a collapsed but finite representation.
        centered = finite(pre_norm - finite(pre_norm.mean(dim=-1, keepdim=True), 'LayerNorm mean'),
                          'LayerNorm centered values')
        finite(centered.square(), 'LayerNorm variance terms')
        result = finite(self.norm(pre_norm), 'LayerNorm').reshape_as(h)
        if diagnostic:
            return result, dict(scores=scores, score_input=score_input, weights=weights,
                                message_input=message_input, messages=messages, aggregate=aggregate,
                                projected=projected, count_term=count_term, pre_norm=pre_norm)
        return result
