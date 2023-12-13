#import necessary libraries
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GAE
from torch_geometric.nn.models.basic_gnn import GAT 
from typing import Optional, List
from torch import Tensor, Union, Callable, Dict, Any
#import OptTensor
from torch_geometric.typing import OptTensor 



# The GIN model
class GNNStack(nn.Module):
    """A 3 layers GNN with GIN convolution and post-message-passing readout layer for graph classification."""
    def __init__(self, input_dim, hidden_dim, output_dim, task='graph'):
        super(GNNStack, self).__init__()
        self.task = task
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25), 
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        else:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.LeakyReLU(), nn.Linear(hidden_dim, hidden_dim)))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
          x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.leaky_relu(x) #to avoid dead ReLU
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)
            emb = x

        x = self.post_mp(x)
        #compute the logits instead of the probabilities
        x = F.log_softmax(x, dim=1)

        return emb, x
    
    def get_activation(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
          x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.leaky_relu(x) #to avoid dead ReLU
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'graph':
            activation = x

        return activation

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    
    def predict(self, data):
        raise NotImplementedError
    #TODO: complete the predict function
    


class GATcustom(GAT):
    """The GAT model from the pytorch geometric library, with a custom post-message-passing FC layer for graph level classification.
    It also implements the possibility to extract embeddings after pooling operation and before the FC layer, as well as
    the possibility to extract the activation of the last convolutional layer."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int, #hidden dimension in convolution
        num_classes: int, #final number of classes at graph level
        num_layers: int,
        out_channels: Optional[int] = None, #output channels of the last convolutional layer
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        **kwargs,
    ):
        super(GATcustom, self).__init__(in_channels, hidden_channels, num_layers, out_channels, dropout, act, act_first, act_kwargs, norm, norm_kwargs, jk, **kwargs)
        self.task = 'graph'
        self.num_classes = num_classes
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.Dropout(0.25), 
            nn.Linear(hidden_channels, self.num_classes)) #solve to have 2 classes , for now out_channels are the number of channels before the FCNN
    def forward(  # noqa
        self,
        data,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
    ):
        r"""
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the underlying GNN layer). (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            num_sampled_nodes_per_hop (List[int], optional): The number of
                sampled nodes per hop.
                Useful in :class:`~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
            num_sampled_edges_per_hop (List[int], optional): The number of
                sampled edges per hop.
                Useful in :class:`~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if (num_sampled_nodes_per_hop is not None
                and isinstance(edge_weight, Tensor)
                and isinstance(edge_attr, Tensor)):
            raise NotImplementedError("'trim_to_layer' functionality does not "
                                      "yet support trimming of both "
                                      "'edge_weight' and 'edge_attr'")

        xs: List[Tensor] = []
        assert len(self.convs) == len(self.norms)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if (num_sampled_nodes_per_hop is not None
                    and not torch.jit.is_scripting()):
                x, edge_index, value = self._trim(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_weight if edge_weight is not None else edge_attr,
                )
                if edge_weight is not None:
                    edge_weight = value
                else:
                    edge_attr = value

            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight and self.supports_edge_attr:
                x = conv(x, edge_index, edge_weight=edge_weight,
                         edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = conv(x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)

            emb = x #save the embeddings of the last layer
            if i < self.num_layers - 1 or self.jk_mode is not None:
                if self.act is not None and self.act_first:
                    x = self.act(x)
                if self.supports_norm_batch:
                    x = norm(x, batch, batch_size)
                else:
                    x = norm(x)
                if self.act is not None and not self.act_first:
                    x = self.act(x)
                x = self.dropout(x)
                if hasattr(self, 'jk'):
                    xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x

        x = pyg_nn.global_mean_pool(x, batch)
        emb = x
        x = self.post_mp(x)
        #compute the logits instead of the probabilities
        x = F.log_softmax(x, dim=1)

        return emb, x
    
    def get_activation(  # noqa
        self,
        data,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
    ):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if (num_sampled_nodes_per_hop is not None
                and isinstance(edge_weight, Tensor)
                and isinstance(edge_attr, Tensor)):
            raise NotImplementedError("'trim_to_layer' functionality does not "
                                      "yet support trimming of both "
                                      "'edge_weight' and 'edge_attr'")

        xs: List[Tensor] = []
        assert len(self.convs) == len(self.norms)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if (num_sampled_nodes_per_hop is not None
                    and not torch.jit.is_scripting()):
                x, edge_index, value = self._trim(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_weight if edge_weight is not None else edge_attr,
                )
                if edge_weight is not None:
                    edge_weight = value
                else:
                    edge_attr = value

            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight and self.supports_edge_attr:
                x = conv(x, edge_index, edge_weight=edge_weight,
                         edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = conv(x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)

            emb = x #save the embeddings of the last layer
            if i < self.num_layers - 1 or self.jk_mode is not None:
                if self.act is not None and self.act_first:
                    x = self.act(x)
                if self.supports_norm_batch:
                    x = norm(x, batch, batch_size)
                else:
                    x = norm(x)
                if self.act is not None and not self.act_first:
                    x = self.act(x)
                x = self.dropout(x)
                if hasattr(self, 'jk'):
                    xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        activation = x

        return activation
        
    def loss(self, pred, label):
        return F.nll_loss(pred, label)