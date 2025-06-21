import random
import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms

 
from torch_geometric.utils import to_networkx

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.nn import Parameter, Linear, ReLU, BatchNorm1d, Module, Sequential

from torch_scatter import scatter

import torch_geometric

from torch_geometric.utils import degree, add_self_loops
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.datasets import QM9


import matplotlib.pyplot as plt

dataset = QM9(root='data/QM9')
trainload = DataLoader(dataset[:10000], batch_size=32, shuffle=True)
testload = DataLoader(dataset[10000:], batch_size=32, shuffle=False)

 
class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim , hidden_layers , edge_dim, aggr='add'):
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        self.mlp_msg = nn.Sequential(
            Linear(2 * emb_dim + edge_dim, hidden_layers),nn.ReLU(), nn.BatchNorm1d(hidden_layers), 
            Linear(hidden_layers, emb_dim), nn.ReLU(), nn.BatchNorm1d(emb_dim)
        )

        self.upd_msg = nn.Sequential(
            Linear(emb_dim * 2, hidden_layers), nn.BatchNorm1d(hidden_layers), nn.ReLU(),
            Linear(hidden_layers, emb_dim), nn.ReLU(), nn.BatchNorm1d(emb_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        self.x = x  
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.mlp_msg(msg_input)

    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=0, reduce=self.aggr)

    def update(self, aggr_out, x):
        upd_input = torch.cat([x, aggr_out], dim=1)
        #print(upd_input.shape)
        return self.upd_msg(upd_input)


class MPNNModel(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim=64, num_layers=3):
        super().__init__()

        #1.
        self.lin_in = Linear(in_dim, hidden_dim)
        #2. layers of calling
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim = hidden_dim,hidden_layers =32, edge_dim = edge_dim))
        #3.Pooling
        self.pool = global_mean_pool
        #4. Linear
        self.lin_layer = nn.Linear(in_features=hidden_dim,
                                   out_features= 1)

    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x =self.lin_in(data.x)

        for conv in self.convs:
            x += conv(x, edge_index, edge_attr)

        #print(x)
        x = self.pool(x, batch)

        #print(self.lin_layer(x).shape)
        return self.lin_layer(x)

model_1 = MPNNModel(in_dim=dataset.num_node_features, edge_dim=dataset.num_edge_features)

optimizer = torch.optim.Adam(model_1.parameters(), lr = 0.01, weight_decay=5e-4)
loss = nn.MSELoss()

def train(train_loader, model, optimizer, loss_fn):
    model_1.train()
    train_loss = 0
    
    for data in train_loader:
        #print(f"LENTH OF X: {len(data.x)}")
        y_pred = model(data)
        #print(f" Prediction: {y_pred.squeeze().shape}| Truth: {data.y.shape}")
        loss = loss_fn(y_pred.squeeze(), data.y[:,4])
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(train_loader)
    
    return train_loss

def test(test_loader, model, loss_fn):
    model_1.eval()
    test_loss = 0
    with torch.inference_mode():
        for data in test_loader:
            test_pred = model(data)
            #print(test_pred)
            loss = loss_fn(test_pred.squeeze(), data.y[:,4])
            test_loss += loss.item()

        test_loss /= len(test_loader)

    return test_loss

ep = []
result1 = []
result2 = []

for epoch in range (5):
    train_loss = train(trainload, model_1, optimizer, loss)
    test_loss = test( testload,model_1, loss)
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



