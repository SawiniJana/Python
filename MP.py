import torch
import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree, scatter, index_to_mask
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root = "/tmp/Cora", name= "Cora", split="random", )
data = dataset[0]

n = dataset[0].num_nodes

train_ratio = int(0.6 * n)
test_ratio = int(0.2 * n) + train_ratio 

perm = torch.randperm(n)
train_idx = perm[:train_ratio]            #tensor
test_idx = perm[train_ratio:test_ratio]   #tensor
val_idx = perm[test_ratio:]               #tensor

data.train_mask = index_to_mask(train_idx, n)     #indices **bool** **tensor**
data.val_mask = index_to_mask(val_idx, size=n)    #indices
data.test_mask = index_to_mask(test_idx, size=n)  #indices


class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim , hidden_layers , edge_dim, aggr='add'):
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        self.mlp_msg = nn.Sequential(
            Linear(2 * emb_dim + edge_dim, hidden_layers), nn.BatchNorm1d(hidden_layers), nn.ReLU(),nn.Dropout(0.5),
            Linear(hidden_layers, emb_dim), nn.ReLU(),nn.Dropout(0.5), nn.BatchNorm1d(emb_dim)
        )

        self.upd_msg = nn.Sequential(
            Linear(2 * emb_dim + edge_dim, hidden_layers), nn.BatchNorm1d(hidden_layers), nn.ReLU(),nn.Dropout(0.5),
            Linear(hidden_layers, emb_dim), nn.ReLU(),nn.Dropout(0.5), nn.BatchNorm1d(emb_dim)
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
                                   out_features= 7)

    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x =self.lin_in(data.x)

        for conv in self.convs:
            x += conv(x, edge_index, edge_attr)

        #print(x.shape)
        #x = self.pool(x, batch) **I DON'T Require to do this step since Cora has just one graph.. otherwise we get [1,64]
        #which is an error**

        #print(self.lin_layer(x).shape)
        return self.lin_layer(x)

num_edges = data.edge_index.size(1)
data.edge_attr = torch.ones((num_edges, 10))
#print(data.edge_attr.size(1))

model = MPNNModel(in_dim = data.x.size(1), edge_dim = data.edge_attr.size(1))        

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=5e-4)
loss = nn.CrossEntropyLoss()

#print(data.train_mask.nonzero(as_tuple=True)[0])

def train(model, data, loss_fn, optimizer):
    model.train()
    optimizer.zero_grad()
    y_pred = model(data)
    
    #print(data.y[data.train_mask].dtype) #Since the targets are stored as int64(Long) need to convert them to float32 for 
                                        # loss.backward() to work
    loss = loss_fn(y_pred[data.train_mask],
                   data.y[data.train_mask])
    #print(loss)
    loss.backward()
    optimizer.step()

    return loss.item()

def test(model, data, loss_fn):
    model.eval()
    y_pred = model(data)
    
    loss = loss_fn(y_pred[data.test_mask],
                   data.y[data.test_mask])
    
    
    return loss.item()

ep = []
result1 = []
result2 = []
test_min = torch.inf

epoch = 200
for i in range(200):
    train_loss = train(model, data, loss, optimizer)
    test_loss = test(model, data, loss)

    ep.append(i)
    result1.append(train_loss)
    result2.append(test_loss)

    if i%20 == 0:
        print(f"Epoch {i} | Train Loss: {train_loss} | Test Loss: {test_loss}")
    if test_loss < test_min:
        test_min = test_loss
        torch.save(model.state_dict(), 'model.pt')

import matplotlib.pyplot as plt
#plotting curves
plt.plot(ep, result1, color = "red", label = "Train_Loss")
plt.plot(ep, result2, color = "blue", label="Test_Loss")
plt.legend()
plt.title("GNN Loss Curves)")
plt.show()

#Saved parameters
#print(t)
#validation
def eval_model(model, data):
    model.eval()
    true_labels = 0

    import numpy as np
    y_pred = model(data)
    #print(y_pred[data.val_mask],data.y[data.val_mask])

    preds = y_pred.argmax(dim=1)
    true_labels += (preds[data.val_mask]==data.y[data.val_mask]).sum()
    print(f"Accuracy: {true_labels/len(val_idx)*100}%")
    

model_test = MPNNModel(in_dim = data.x.size(1), edge_dim = data.edge_attr.size(1))
model_test.load_state_dict(torch.load('model.pt', weights_only=True))
eval_model(model_test, data)