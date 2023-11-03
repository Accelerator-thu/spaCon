# Several GNN models for feature extraction

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    MessagePassing,
    SAGEConv,
    global_add_pool,
)
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.utils import add_self_loops, degree, softmax


class GNN(nn.Module):
    def __init__(self, gnn_type, in_channels, hidden_channels, out_channels, num_layers, dropout, graph_pooling, batch_norm):
        super(GNN, self).__init__()
        
        self.gnn_type = gnn_type
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.graph_pooling = graph_pooling
        self.batch_norm = batch_norm
        
        if self.gnn_type == 'GCN':
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(self.in_channels, self.hidden_channels))
            for i in range(self.num_layers - 2):
                self.convs.append(
                    GCNConv(self.hidden_channels, self.hidden_channels))
            self.convs.append(
                GCNConv(self.hidden_channels, self.out_channels))
            
        elif self.gnn_type == 'GAT':
            self.convs = nn.ModuleList()
            self.convs.append(
                GATConv(self.in_channels, self.hidden_channels, heads=8, dropout=self.dropout))
            for i in range(self.num_layers - 2):
                self.convs.append(
                    GATConv(self.hidden_channels * 8, self.hidden_channels, heads=8, dropout=self.dropout))
            self.convs.append(
                GATConv(self.hidden_channels * 8, self.out_channels, heads=8, dropout=self.dropout))
            
        elif self.gnn_type == 'SAGE':
            self.convs = nn.ModuleList()
            self.convs.append(
                SAGEConv(self.in_channels, self.hidden_channels))
            for i in range(self.num_layers - 2):
                self.convs.append(
                    SAGEConv(self.hidden_channels, self.hidden_channels))
            self.convs.append(
                SAGEConv(self.hidden_channels, self.out_channels))
            
        elif self.gnn_type == 'GIN':
            self.convs = nn.ModuleList()
            self.convs.append(
                GINConv(nn.Sequential(nn.Linear(self.in_channels, self.hidden_channels), 
                                      nn.ReLU(), 
                                      nn.Linear(self.hidden_channels, self.hidden_channels))))
            for i in range(self.num_layers - 2):
                self.convs.append(
                    GINConv(nn.Sequential(nn.Linear(self.hidden_channels, self.hidden_channels), 
                                          nn.ReLU(), 
                                          nn.Linear(self.hidden_channels, self.hidden_channels))))
            self.convs.append(
                GINConv(nn.Sequential(nn.Linear(self.hidden_channels, self.out_channels), 
                                      nn.ReLU(), 
                                      nn.Linear(self.out_channels, self.out_channels))))
            
        if self.batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(self.num_layers):
                self.batch_norms.append(nn.BatchNorm1d(self.hidden_channels))
        
        if self.graph_pooling == 'sum':
            self.pool = global_add_pool
        elif self.graph_pooling == 'mean':
            self.pool = global_mean_pool
        elif self.graph_pooling == 'max':
            self.pool = global_max_pool
        elif self.graph_pooling == 'none':
            self.pool = None
            
    def forward(self, x, edge_index, batch):
        if self.gnn_type == 'GIN':
            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index)
                if self.batch_norm:
                    x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index)
                if self.batch_norm:
                    x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.pool is not None:
            x = self.pool(x, batch)
        
        return x
    
    def momentumUpdateParameters(self, mu, otherparams):
        for param, otherparam in zip(self.parameters(), otherparams):
            param.data = mu * param.data + (1 - mu) * otherparam.data
        
    
