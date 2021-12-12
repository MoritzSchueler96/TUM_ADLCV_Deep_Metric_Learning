import numpy as np
import torch
from torch import nn

"""
---
title: Graph Attention Networks (GAT)
summary: >
 A PyTorch implementation/tutorial of Graph Attention Networks.
---
# Graph Attention Networks (GAT)
This is a [PyTorch](https://pytorch.org) implementation of the paper
[Graph Attention Networks](https://papers.labml.ai/paper/1710.10903).
GATs work on graph data.
A graph consists of nodes and edges connecting nodes.
For example, in Cora dataset the nodes are research papers and the edges are citations that
connect the papers.
GAT uses masked self-attention, kind of similar to [transformers](../../transformers/mha.html).
GAT consists of graph attention layers stacked on top of each other.
Each graph attention layer gets node embeddings as inputs and outputs transformed embeddings.
The node embeddings pay attention to the embeddings of other nodes it's connected to.
The details of graph attention layers are included alongside the implementation.
Here is [the training code](experiment.html) for training
a two-layer GAT on Cora dataset.
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/d6c636cadf3511eba2f1e707f612f95d)
"""


class GraphAttentionLayer(nn.Module):
    """
    ## Graph attention layer
    This is a single graph attention layer.
    A GAT is made up of multiple such layers.
    It takes
    $$\mathbf{h} = \{ \overrightarrow{h_1}, \overrightarrow{h_2}, \dots, \overrightarrow{h_N} \}$$,
    where $\overrightarrow{h_i} \in \mathbb{R}^F$ as input
    and outputs
    $$\mathbf{h'} = \{ \overrightarrow{h'_1}, \overrightarrow{h'_2}, \dots, \overrightarrow{h'_N} \}$$,
    where $\overrightarrow{h'_i} \in \mathbb{R}^{F'}$.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        is_concat: bool = True,
        dropout: float = 0.6,
        leaky_relu_negative_slope: float = 0.2,
    ):
        """
        * `in_features`, $F$, is the number of input features per node
        * `out_features`, $F'$, is the number of output features per node
        * `n_heads`, $K$, is the number of attention heads
        * `is_concat` whether the multi-head results should be concatenated or averaged
        * `dropout` is the dropout probability
        * `leaky_relu_negative_slope` is the negative slope for leaky relu activation
        """
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial transformation;
        # i.e. to transform the node embeddings before self-attention
        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `h`, $\mathbf{h}$ is the input node embeddings of shape `[n_nodes, in_features]`.
        * `adj_mat` is the adjacency matrix of shape `[n_nodes, n_nodes, n_heads]`.
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.
        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`.
        """

        # Number of nodes
        n_nodes = h.shape[0]
        # We do single linear transformation and then split it up for each head.
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)

        # #### Calculate attention score
        g_repeat = g.repeat(n_nodes, 1, 1)
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        # Now we concatenate
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        # Reshape
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)

        # Calculate
        e = self.activation(self.attn(g_concat))
        # Remove the last dimension of size `1`
        e = e.squeeze(-1)

        # The adjacency matrix should have shape
        # `[n_nodes, n_nodes, n_heads]` or`[n_nodes, n_nodes, 1]`
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        # Mask $e_{ij}$ based on adjacency matrix.
        e = e.masked_fill(adj_mat == 0, float("-inf"))

        # We then normalize attention scores (or coefficients)
        a = self.softmax(e)

        # Apply dropout regularization
        a = self.dropout(a)

        # Calculate final output for each head
        attn_res = torch.einsum("ijh,jhf->ihf", a, g)

        # Concatenate the heads
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            return attn_res.mean(dim=1)


class GraphAttentionV2Layer(nn.Module):
    """
    ## Graph attention v2 layer
    This is a single graph attention v2 layer.
    A GATv2 is made up of multiple such layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        is_concat: bool = True,
        dropout: float = 0.6,
        leaky_relu_negative_slope: float = 0.2,
        share_weights: bool = False,
    ):
        """
        * `in_features`, $F$, is the number of input features per node
        * `out_features`, $F'$, is the number of output features per node
        * `n_heads`, $K$, is the number of attention heads
        * `is_concat` whether the multi-head results should be concatenated or averaged
        * `dropout` is the dropout probability
        * `leaky_relu_negative_slope` is the negative slope for leaky relu activation
        * `share_weights` if set to `True`, the same matrix will be applied to the source and the target node of every edge
        """
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial source transformation;
        # i.e. to transform the source node embeddings before self-attention
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # If `share_weights` is `True` the same linear layer is used for the target nodes
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `h`, $\mathbf{h}$ is the input node embeddings of shape `[n_nodes, in_features]`.
        * `adj_mat` is the adjacency matrix of shape `[n_nodes, n_nodes, n_heads]`.
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.
        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`.
        """

        # Number of nodes
        n_nodes = h.shape[0]
        # The initial transformations for each head.
        # We do two linear transformations and then split it up for each head.
        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_nodes, self.n_heads, self.n_hidden)

        # #### Calculate attention score
        g_l_repeat = g_l.repeat(n_nodes, 1, 1)
        # each node embedding is repeated `n_nodes` times.
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)
        # Now we add the two tensors
        g_sum = g_l_repeat + g_r_repeat_interleave
        # Reshape
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)

        # Calculate attention
        e = self.attn(self.activation(g_sum))
        # Remove the last dimension of size `1`
        e = e.squeeze(-1)

        # The adjacency matrix should have shape
        # `[n_nodes, n_nodes, n_heads]` or`[n_nodes, n_nodes, 1]`
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        # Mask $e_{ij}$ based on adjacency matrix.
        e = e.masked_fill(adj_mat == 0, float("-inf"))

        # We then normalize attention scores (or coefficients)
        a = self.softmax(e)

        # Apply dropout regularization
        a = self.dropout(a)

        # Calculate final output for each head
        attn_res = torch.einsum("ijh,jhf->ihf", a, g_r)

        # Concatenate the heads
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            return attn_res.mean(dim=1)


class GAT(nn.Module):
    """
    ## Graph Attention Network (GAT)
    This graph attention network has two [graph attention layers](index.html).
    """

    def __init__(
        self,
        in_features: int,
        n_hidden: int,
        n_classes: int,
        n_heads: int,
        dropout: float,
    ):
        """
        * `in_features` is the number of features per node
        * `n_hidden` is the number of features in the first graph attention layer
        * `n_classes` is the number of classes
        * `n_heads` is the number of heads in the graph attention layers
        * `dropout` is the dropout probability
        """
        super().__init__()

        # First graph attention layer where we concatenate the heads
        self.layer1 = GraphAttentionLayer(
            in_features, n_hidden, n_heads, is_concat=True, dropout=dropout
        )
        # Activation function after first graph attention layer
        self.activation = nn.ELU()
        # Final graph attention layer where we average the heads
        self.output = GraphAttentionLayer(
            n_hidden, n_classes, 1, is_concat=False, dropout=dropout
        )
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `x` is the features vectors of shape `[n_nodes, in_features]`
        * `adj_mat` is the adjacency matrix of the form
         `[n_nodes, n_nodes, n_heads]` or `[n_nodes, n_nodes, 1]`
        """
        # Apply dropout to the input
        x = self.dropout(x)
        # First graph attention layer
        x = self.layer1(x, adj_mat)
        # Activation function
        x = self.activation(x)
        # Dropout
        x = self.dropout(x)
        # Output layer (without activation) for logits
        return self.output(x, adj_mat)


class GATv2(nn.Module):
    """
    ## Graph Attention Network v2 (GATv2)
    This graph attention network has two [graph attention layers](index.html).
    """

    def __init__(
        self,
        in_features: int,
        n_hidden: int,
        n_classes: int,
        n_heads: int,
        dropout: float,
        share_weights: bool = True,
    ):
        """
        * `in_features` is the number of features per node
        * `n_hidden` is the number of features in the first graph attention layer
        * `n_classes` is the number of classes
        * `n_heads` is the number of heads in the graph attention layers
        * `dropout` is the dropout probability
        * `share_weights` if set to True, the same matrix will be applied to the source and the target node of every edge
        """
        super().__init__()

        # First graph attention layer where we concatenate the heads
        self.layer1 = GraphAttentionV2Layer(
            in_features,
            n_hidden,
            n_heads,
            is_concat=True,
            dropout=dropout,
            share_weights=share_weights,
        )
        # Activation function after first graph attention layer
        self.activation = nn.ELU()
        # Final graph attention layer where we average the heads
        self.output = GraphAttentionV2Layer(
            n_hidden,
            n_classes,
            1,
            is_concat=False,
            dropout=dropout,
            share_weights=share_weights,
        )
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `x` is the features vectors of shape `[n_nodes, in_features]`
        * `adj_mat` is the adjacency matrix of the form
         `[n_nodes, n_nodes, n_heads]` or `[n_nodes, n_nodes, 1]`
        """
        # Apply dropout to the input
        x = self.dropout(x)
        # First graph attention layer
        x = self.layer1(x, adj_mat)
        # Activation function
        x = self.activation(x)
        # Dropout
        x = self.dropout(x)
        # Output layer (without activation) for logits
        return self.output(x, adj_mat)
