from torch_scatter import scatter_mean, scatter_max, scatter_add
import torch
from torch import _shape_as_tensor, nn
import math
import torch.nn.functional as F
import numpy as np
import torch
import logging

from torch_geometric.nn.conv import GATConv, GATv2Conv

from .utils import *
from .attentions import MultiHeadDotProduct
import torch.utils.checkpoint as checkpoint

logger = logging.getLogger("GNNReID.GNNModule")


class MetaLayer(torch.nn.Module):
    """
        Core Message Passing Network Class. Extracted from torch_geometric, with minor modifications.
        (https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/meta.py)
    """

    def __init__(self, edge_model=None, node_model=None):
        super(MetaLayer, self).__init__()
        self.edge_model = edge_model  # possible to add edge model
        self.node_model = node_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, feats, edge_index, edge_attr=None):

        r, c = edge_index[:, 0], edge_index[:, 1]

        if self.edge_model is not None:
            edge_attr = torch.cat([feats[r], feats[c], edge_attr], dim=1)
            edge_attr = self.edge_model(edge_attr)

        if self.node_model is not None:
            feats, edge_index, edge_attr = self.node_model(feats, edge_index, edge_attr)

        return feats, edge_index, edge_attr

    def __repr__(self):
        if self.edge_model:
            return ("{}(\n" "    edge_model={},\n" "    node_model={},\n" ")").format(
                self.__class__.__name__, self.edge_model, self.node_model
            )
        else:
            return ("{}(\n" "    node_model={},\n" ")").format(self.__class__.__name__, self.node_model)

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
        e = e.masked_fill(adj_mat.to("cuda") == 0, float("-inf"))

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

class GNNReID(nn.Module):
    def __init__(self, dev, n_nodes: int = 70, params: dict = None, embed_dim: int = 2048):
        super(GNNReID, self).__init__()
        num_classes = params["classifier"]["num_classes"]
        self.dev = dev
        self.params = params
        self.gnn_params = params["gnn"]
        self.gat = "gat" in self.gnn_params

        self.dim_red = nn.Linear(embed_dim, int(embed_dim / params["red"]))
        logger.info("Embed dim old {}".format(embed_dim))
        embed_dim = int(embed_dim / params["red"])
        logger.info("Embed dim new {}".format(embed_dim))

        self.adj_mat = torch.ones((n_nodes, n_nodes, 1))
        self.gnn_model = self._build_GNN_Net(self.adj_mat, embed_dim=embed_dim)

        # classifier
        self.neck = params["classifier"]["neck"]
        dim = self.gnn_params["num_layers"] * embed_dim if self.params["cat"] else embed_dim
        every = self.params["every"]
        if self.neck:
            layers = (
                [nn.BatchNorm1d(dim) for _ in range(self.gnn_params["num_layers"])] if every else [nn.BatchNorm1d(dim)]
            )
            self.bottleneck = Sequential(*layers)
            for layer in self.bottleneck:
                layer.bias.requires_grad_(False)
                layer.apply(weights_init_kaiming)

            layers = (
                [nn.Linear(dim, num_classes, bias=False) for _ in range(self.gnn_params["num_layers"])]
                if every
                else [nn.Linear(dim, num_classes, bias=False)]
            )
            self.fc = Sequential(*layers)
            print(self.fc)
            for layer in self.fc:
                layer.apply(weights_init_classifier)
        else:
            layers = (
                [nn.Linear(dim, num_classes) for _ in range(self.gnn_params["num_layers"])]
                if every
                else [nn.Linear(dim, num_classes)]
            )
            self.fc = Sequential(*layers)

    def _build_GNN_Net(self, adj_mat: torch.tensor, embed_dim: int = 2048):
        if self.gat:
            gnn_model = GATNetwork(adj_mat, embed_dim, self.gnn_params, self.gnn_params["num_layers"])
        else:
            # init aggregator
            if self.gnn_params["aggregator"] == "add":
                self.aggr = lambda out, row, dim, x_size: scatter_add(out, row, dim=dim, dim_size=x_size)
            if self.gnn_params["aggregator"] == "mean":
                self.aggr = lambda out, row, dim, x_size: scatter_mean(out, row, dim=dim, dim_size=x_size)
            if self.gnn_params["aggregator"] == "max":
                self.aggr = lambda out, row, dim, x_size: scatter_max(out, row, dim=dim, dim_size=x_size)

            gnn = GNNNetwork(embed_dim, self.aggr, self.dev, self.gnn_params, self.gnn_params["num_layers"])
            gnn_model = MetaLayer(node_model=gnn)

        return gnn_model

    def forward(self, feats, adj_mat, edge_index, edge_attr=None, output_option="norm"):
        r, c = edge_index[:, 0], edge_index[:, 1]

        if self.dim_red is not None:
            feats = self.dim_red(feats)

        if self.gat:
            edge_index = edge_index.t()
            feats = self.gnn_model(feats, adj_mat, edge_index, edge_attr)
        else:
            feats, _, _ = self.gnn_model(feats, edge_index, edge_attr)

        if self.params["cat"]:
            feats = [torch.cat(feats, dim=1).to(self.dev)]
        elif self.params["every"]:
            feats = feats
        else:
            feats = [feats[-1]]

        if self.neck:
            features = list()
            for i, layer in enumerate(self.bottleneck):
                f = layer(feats[i])
                features.append(f)
        else:
            features = feats

        x = list()
        for i, layer in enumerate(self.fc):
            f = layer(features[i])
            x.append(f)

        if output_option == "norm":
            return x, feats
        elif output_option == "plain":
            return x, [F.normalize(f, p=2, dim=1) for f in feats]
        elif output_option == "neck" and self.neck:
            return x, features
        elif output_option == "neck" and not self.neck:
            print("Output option neck only avaiable if bottleneck (neck) is " "enabled - giving back x and fc7")
            return x, feats

        return x, feats


class GNNNetwork(nn.Module):
    def __init__(self, embed_dim, aggr, dev, gnn_params, num_layers):
        super(GNNNetwork, self).__init__()

        layers = [DotAttentionLayer(embed_dim, aggr, dev, gnn_params) for _ in range(num_layers)]

        self.layers = Sequential(*layers)

    def forward(self, feats, edge_index, edge_attr):
        out = list()
        for layer in self.layers:
            feats, edge_index, edge_attr = layer(feats, edge_index, edge_attr)
            out.append(feats)
        return out, edge_index, edge_attr


class DotAttentionLayer(nn.Module):
    def __init__(self, embed_dim, aggr, dev, params, d_hid=None):
        super(DotAttentionLayer, self).__init__()
        num_heads = params["num_heads"]
        self.res1 = params["res1"]
        self.res2 = params["res2"]

        self.att = MultiHeadDotProduct(embed_dim, num_heads, aggr, mult_attr=params["mult_attr"]).to(dev)

        d_hid = 4 * embed_dim if d_hid is None else d_hid
        self.mlp = params["mlp"]

        self.linear1 = nn.Linear(embed_dim, d_hid) if params["mlp"] else None
        self.dropout = nn.Dropout(params["dropout_mlp"])
        self.linear2 = nn.Linear(d_hid, embed_dim) if params["mlp"] else None

        self.norm1 = LayerNorm(embed_dim) if params["norm1"] else None
        self.norm2 = LayerNorm(embed_dim) if params["norm2"] else None
        self.dropout1 = nn.Dropout(params["dropout_1"])
        self.dropout2 = nn.Dropout(params["dropout_2"])

        self.act = F.relu

        self.dummy_tensor = torch.ones(1, requires_grad=True)

    def custom(self):
        def custom_forward(*inputs):
            feats2 = self.att(inputs[0], inputs[1], inputs[2])
            return feats2

        return custom_forward

    def forward(self, feats, edge_index, edge_attr):
        feats2 = self.att(feats, edge_index, edge_attr)
        # if gradient checkpointing should be apllied for the gnn, comment line above and uncomment line below
        # feats2 = checkpoint.checkpoint(self.custom(), feats, edge_index, edge_attr, preserve_rng_state=True)

        feats2 = self.dropout1(feats2)
        feats = feats + feats2 if self.res1 else feats2
        feats = self.norm1(feats) if self.norm1 is not None else feats

        if self.mlp:
            feats2 = self.linear2(self.dropout(self.act(self.linear1(feats))))
        else:
            feats2 = feats

        feats2 = self.dropout2(feats2)
        feats = feats + feats2 if self.res2 else feats2
        feats = self.norm2(feats) if self.norm2 is not None else feats

        return feats, edge_index, edge_attr


class GATNetwork(nn.Module):
    def __init__(self, adj_mat, embed_dim, params, num_layers):
        super(GATNetwork, self).__init__()
        self.res1 = params["res1"]
        self.res2 = params["res2"]

        self.adj_mat = adj_mat

        layers = list()
        lin_layers = list()

        if params["gat"] == 1:
            for _ in range(num_layers):
                layers.append(
                    GATConv(
                        in_channels=embed_dim,
                        out_channels=embed_dim,
                        heads=params["num_heads"],
                        concat=False,
                        dropout=params["dropout_gat"],
                        add_self_loops=False,
                        edge_dim=1,
                    ))
                lin_layers.append(LinearLayer(embed_dim, params))

        elif params["gat"] == 2:
            for _ in range(num_layers):
                layers.append(
                    GATv2Conv(
                        in_channels=embed_dim,
                        out_channels=embed_dim,
                        heads=params["num_heads"],
                        concat=False,
                        dropout=params["dropout_gat"],
                        add_self_loops=False,
                        edge_dim=1,
                    ))
                lin_layers.append(LinearLayer(embed_dim, params))

        elif params["gat"] == 3:
            for _ in range(num_layers):
                layers.append(
                    GAT(
                        in_features=embed_dim,
                        n_hidden=embed_dim,
                        n_classes=embed_dim,
                        n_heads=params["num_heads"],
                        dropout=params["dropout_gat"],
                    ))
                lin_layers.append(LinearLayer(embed_dim, params))

        elif params["gat"] == 4:
            for _ in range(num_layers):
                layers.append(
                    GATv2(
                        in_features=embed_dim,
                        n_hidden=embed_dim,
                        n_classes=embed_dim,
                        n_heads=params["num_heads"],
                        dropout=params["dropout_gat"],
                    ))
                lin_layers.append(LinearLayer(embed_dim, params))

        self.layers = Sequential(*layers)
        self.lin_layers = Sequential(*lin_layers)

    def forward(self, feats, adj_mat, edge_index, edge_attr):
        out = list()
        for layer, lin_layer in zip(self.layers, self.lin_layers):
            feats = layer(feats, adj_mat)
            feats, _, _ = lin_layer(feats, edge_index, edge_attr)
            out.append(feats)
        return out


class LinearLayer(nn.Module):
    def __init__(self, embed_dim, params, d_hid=None):
        super(LinearLayer, self).__init__()
        self.res1 = params["res1"]
        self.res2 = params["res2"]

        d_hid = 4 * embed_dim if d_hid is None else d_hid
        self.mlp = params["mlp"]

        self.linear1 = nn.Linear(embed_dim, d_hid) if params["mlp"] else None
        self.dropout = nn.Dropout(params["dropout_mlp"])
        self.linear2 = nn.Linear(d_hid, embed_dim) if params["mlp"] else None

        self.norm1 = LayerNorm(embed_dim) if params["norm1"] else None
        self.norm2 = LayerNorm(embed_dim) if params["norm2"] else None
        self.dropout1 = nn.Dropout(params["dropout_1"])
        self.dropout2 = nn.Dropout(params["dropout_2"])

        self.act = F.relu


    def forward(self, feats, edge_index, edge_attr):
        # if gradient checkpointing should be apllied for the gnn, comment line above and uncomment line below
        # feats2 = checkpoint.checkpoint(self.custom(), feats, edge_index, edge_attr, preserve_rng_state=True)

        feats2 = self.dropout1(feats)
        feats = feats + feats2 if self.res1 else feats2
        feats = self.norm1(feats) if self.norm1 is not None else feats

        if self.mlp:
            feats2 = self.linear2(self.dropout(self.act(self.linear1(feats))))
        else:
            feats2 = feats

        feats2 = self.dropout2(feats2)
        feats = feats + feats2 if self.res2 else feats2
        feats = self.norm2(feats) if self.norm2 is not None else feats

        return feats, edge_index, edge_attr
