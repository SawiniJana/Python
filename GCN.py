import torch
import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')

Size = 32

train_ratio = 0.6
test_ratio = 0.2
valid_ratio = 0.2

torch.manual_seed(42)
train_set, test_set, valid_test = random_split(dataset, [train_ratio, test_ratio, valid_ratio])
train_load = DataLoader(train_set,
                        shuffle=True,
                        batch_size=Size)
test_load = DataLoader(train_set,
                        shuffle=False,
                        batch_size=Size)
valid_load = DataLoader(train_set,
                        shuffle=False,
                        batch_size=Size)

class GCNConv(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="add")
        self.lin = Linear(in_channels, out_channels, bias =False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, batch):
        #x --> [N, in_channels]
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0)) #2,E
        
        x = self.lin(x) # x --> [N, out_channels]

        source, target = edge_index #[E]
        #print(f"HI: {len(target)}")
        deg = degree(target, x.size(0), dtype=x.dtype) #[N]
        deg_inv_sqrt = deg.pow(-0.5) #[N]
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[source] * deg_inv_sqrt[target] #[E]
        #print(f"HI: {norm.shape}")
        out = self.propagate(edge_index, x=x, norm=norm)
        #[N, out_channels]
        #but why change from E to N? 

        out = out + self.bias
        
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
           #*[E,1] x [E, out_channels] = [E, out_channels] 

class GCNModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim,hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        #print("x:", x.shape)
        x = self.conv1(x, edge_index, batch)
        #print("After conv1:", x.shape)
        x = F.relu(x)
        x = self.conv2(x, edge_index, batch)
        #print("After conv2:", x.shape)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        #print("After global mean pool:", x.shape)
        
        return self.lin(x)


model = GCNModule(in_dim= dataset.x.size(1), hidden_dim=64, out_dim= 2)
#5 nodes, 8 edges .. kind of like aggregate collects messages to target nodes
#**message() returns shape [8, 64] → one 64-dim vector per edge
#**aggregate() reduces these into shape [5, 64] → one vector per node

#loss and optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
loss = nn.CrossEntropyLoss()


def train(train_loader, model, optimizer, loss_fn):
    model.train()
    train_loss = 0
    
    for data in train_loader:
        #print(f"LENTH OF X: {len(data.x)}")
        y_pred = model(data.x, data.edge_index, data.batch)
        #print(f" Prediction: {y_pred.squeeze()}| Truth: {data.y}")
        loss = loss_fn(y_pred, data.y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(train_loader)
    
    return train_loss

def test(test_loader, model, loss_fn):
    model.eval()
    test_loss = 0
    with torch.inference_mode():
        for data in test_loader:
            test_pred = model(data.x, data.edge_index, data.batch)
            loss = loss_fn(test_pred, data.y)
            test_loss += loss.item()

        test_loss /= len(test_loader)

    return test_loss

def eval_model(model,valid_loader ):
    model.eval()
    true_reds = num_preds = 0

    with torch.inference_mode():
        for data in valid_loader:
            Valid_pred = model(data.x, data.edge_index, data.batch)
            pred_abels = torch.argmax(Valid_pred, dim= 1)
            print(f"Prediction{pred_abels.squeeze()}|Truth{data.y}")
            true_reds += (pred_abels == data.y).sum()
            num_preds += len(data.y.squeeze())
    
    acc2 = true_reds/num_preds

    return acc2

epoch = 500
ep = []
result1 = []
result2 = []
test_min = torch.inf

for i in range(epoch):
    train_loss = train(train_load, model, optimizer, loss)
    test_loss = test(test_load, model, loss)

    ep.append(i)
    result1.append(train_loss)
    result2.append(test_loss)

    if i%50 == 0:
        print(f"Epoch {i} | Train Loss: {train_loss} | Test Loss: {test_loss}")
    if test_loss < test_min:
        test_min = test_loss
        torch.save(model.state_dict(), 'model.pt')

#Plotting the curves
import matplotlib.pyplot as plt

plt.plot(ep, result1, color = "red", label = "Train_Loss")
plt.plot(ep, result2, color = "blue", label="Test_Loss")
plt.legend()
plt.title("GCN Loss Curves)")
plt.show()

model_test = GCNModule(in_dim= dataset.x.size(1), hidden_dim=64, out_dim= 2)
model_test.load_state_dict(torch.load('model.pt', weights_only=True))
acc2 = eval_model(model_test, valid_load)
print(f"Accuracy of the model: {100.0*acc2}%")



