import os
import time
import random
import numpy as np

from scipy.stats import ortho_group

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch_geometric.transforms
 
from torch_geometric.utils import to_networkx

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.nn import Parameter, Linear, ReLU, BatchNorm1d, Module, Sequential
from torch import Tensor
from torch_scatter import scatter
from torch_cluster import knn


torch.set_default_dtype(torch.float32)

from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)

import networkx as nx
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import Batch
import torch_geometric.transforms as T
from torch_geometric.utils import degree, add_self_loops, remove_self_loops, to_dense_adj, dense_to_sparse, to_undirected, from_networkx, to_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, knn_graph
from torch_geometric.datasets import QM9


import matplotlib.pyplot as plt

from IPython.display import HTML



print("PyTorch version {}".format(torch.__version__))
print("PyG version {}".format(torch_geometric.__version__))

Batch_size = 32

dataset = QM9(root='data/QM9')
trainload = DataLoader(dataset[:10000], batch_size=32, shuffle=True)
testload = DataLoader(dataset[10000:], batch_size=32, shuffle=False)

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="add")
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels)) 

        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()
    
    def forward(self, x , edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes =x.size(0) )
        x = self.lin(x)
        row,col = edge_index
        deg = degree(col, x.size(0), dtype = x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("int")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, norm=norm)
        out += self.bias

        return out
        
    
    def messgae(self, x_j, norm):
        return norm.view(-1,1) * x_j
    
class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr="add"):
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        self.mlp_upd = Sequential(
            Linear(3*emb_dim + edge_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        )

        self.mlp_upd = Sequential(
            Linear(3*emb_dim + edge_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        )

    def forward(self, h, edge_index, edge_attr):
        out = self.propagate(edge_index= edge_index, h=h, edge_attr=edge_attr)
        return out
        
    def message(self, h_i, h_j, edge_attr):    
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return msg
    
    def aggregate(self, inputs, index):
        out = scatter(inputs, index, dim=0, reduce=self.aggr)
        return out
    
    def update(self, aggr_out, h ):
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)
    
class MPNNModel(Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1):

        super(MPNNModel, self).__init__()
        
        # Linear projection for initial node features
        self.lin_in = Linear(in_dim, emb_dim)
        
        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))
        
        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool
        self.lin_pred = Linear(emb_dim, out_dim)
        
    def forward(self, data):
        h = self.lin_in(data.x) # (n, d_n) -> (n, d)
        
        for conv in self.convs:
            h = h + conv(h = h, edge_index = data.edge_index, edge_attr = data.edge_attr) # (n, d) -> (n, d)
            # Note that we add a residual connection after each MPNN layer

 
        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)
        out = self.lin_pred(h_graph) # (batch_size, d) -> (batch_size, 1)
        return out.view(-1)


model_1 = MPNNModel(in_dim=dataset.num_node_features, edge_dim=dataset.num_edge_features)
model_2 = GCNConv(in_channels=dataset.num_node_features, out_channels=64)

optimizer = torch.optim.Adam(model_1.parameters(), lr = 0.0001)
loss = nn.MSELoss()

def train(model_1, trainload):
    model_1.train()
    total_loss = 0
    for data in trainload:
        optimizer.zero_grad()
        out  =model_1(data)
        train_loss = loss(out, data.y[:,0])
        train_loss.backward()
        optimizer.step()
        total_loss += train_loss.item() * data.num_graphs
    
    return total_loss/len(trainload.dataset)

def test(model_1, testload):
    model_1.eval()
    test_loss = 0
    for data in testload:
        
        out  =model_1(data)
        test_loss = loss(out, data.y[:,0])
        test_loss += test_loss.item() * data.num_graphs
    
    return test_loss/len(testload.dataset)

ep = []
result1 = []
result2 = []

for epoch in range (0, 25):
    train_loss = train(model_1,trainload)
    test_loss = test(model_1, testload)
    ep.append(epoch)
    result1.append(np.array(torch.tensor(train_loss).numpy()))
    result2.append(np.array(torch.tensor(test_loss).numpy()))
    
    #draw_epoch_graph(G, model_1.latest)
    print(f"Epoch {epoch} | Train Loss: {train_loss} | Test Loss: {test_loss}")


#plotting loss curves
plt.plot(ep, result1, color = "red", label = "Train_Loss")
plt.plot(ep, result2, color = "blue", label="Test_Loss")
plt.legend()
plt.title("GNN Loss Curves)")
plt.show()



