from logging import raiseExceptions
from os import defpath

from torch.cuda.memory import caching_allocator_alloc
from torch_scatter.scatter import scatter_mean
from .utils import *
from torch import nn
import torch
import math

from einops import rearrange, reduce


class MultiHeadDotProduct(nn.Module):
    """
    Multi head attention like in transformers
    embed_dim: dimension of input embedding
    nhead: number of attention heads
    """

    def __init__(self, embed_dim, nhead, aggr, dropout=0.1, mult_attr=0):
        super(MultiHeadDotProduct, self).__init__()
        print("MultiHeadDotProduct")
        self.embed_dim = embed_dim
        self.hdim = embed_dim // nhead
        self.nhead = nhead
        self.mult_attr = mult_attr

        aggr = "sum" if aggr == "add" else aggr
        assert aggr in ["sum", "max", "mean", "prod"]
        self.aggr = aggr

        # FC Layers for input
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # fc layer for concatenated output
        self.out = nn.Linear(embed_dim, embed_dim)

        self.reset_parameters()

    def forward(self, feats: torch.tensor, edge_index: torch.tensor, edge_attr: torch.tensor):
        q = k = v = feats
        bs = q.size(0)

        # FC layer
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)

        """
        Variables used here:
            h = nheads
            d = hdim
            n = batch_size = nodes
            a = adjacent nodes
        """

        # Split into heads --> h * n * d
        q, k, v = map(lambda t: rearrange(t, "n (h d) -> h n d", h=self.nhead), (q, k, v))

        # Extend according to edges
        r, c, e = edge_index[:, 0], edge_index[:, 1], edge_index.shape[0]
        head_indices = torch.arange(self.nhead).type(torch.cuda.LongTensor).view(self.nhead, 1).expand(-1, e)
        q = q[head_indices, c, :]
        k = k[head_indices, r, :]
        v = v[head_indices, r, :]

        # Adjust dims
        q, k = map(lambda t: rearrange(t, "h (n a) d-> h n a d", h=self.nhead, n=bs), (q, k))

        # Calculate similarity: (Q @ K) / sqrt(d)
        sim = torch.einsum("h n a d , h n a d -> h n a", q, k) / math.sqrt(self.hdim)

        # Attention scores
        # attn = torch.softmax(sim, dim=-2)
        # attn = rearrange(attn, "h n a-> h (n a)")
        sim = rearrange(sim, "h n a-> h (n a)")
        attn = deterministic_softmax(sim, c, bs)

        # Dropout
        attn = self.dropout(attn)

        # Obtain feature matrix and aggregate accordingly
        feats = attn.unsqueeze(-1) * v
        feats = rearrange(feats, "h (n a) d-> h n a d", h=self.nhead, n=bs)

        # Aggregate over each node according to aggregator
        feats = reduce(feats, "h n a d-> h a d", self.aggr)
        feats = rearrange(feats, "h n d -> n (h d)")

        # Linear layer
        feats = self.out(feats)

        return feats  # , edge_index, edge_attr

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.constant_(self.q_linear.bias, 0.0)

        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.constant_(self.v_linear.bias, 0.0)

        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.constant_(self.k_linear.bias, 0.0)

        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.0)
