import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import torch

import torch.nn as nn
from torch.nn import Linear
from torch.utils.data.sampler import SubsetRandomSampler

from torch_geometric.utils import scatter
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.loader import DataLoader

#Device Agnostic code
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

dataset = torch.load("D:\\cleaned_graph_data_electron.pt")

#print(dataset[0])

class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim , hidden_layers , edge_dim, aggr='add'):   #(64, 32, 1)
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim      #64
        self.edge_dim = edge_dim    #1
        
        self.mlp_msg = nn.Sequential(
            Linear(2 * emb_dim + edge_dim, hidden_layers), nn.BatchNorm1d(hidden_layers), nn.ReLU(),nn.Dropout(0.5),
            Linear(hidden_layers, emb_dim), nn.ReLU(),nn.Dropout(0.5), nn.BatchNorm1d(emb_dim)
        )    #(E, 64)

        self.upd_msg = nn.Sequential(
            Linear(emb_dim * 2, hidden_layers), nn.BatchNorm1d(hidden_layers), nn.ReLU(),
            Linear(hidden_layers, emb_dim), nn.ReLU(), nn.BatchNorm1d(emb_dim)
        )   #(N, 64)

    def forward(self, x, edge_index, edge_attr):
        self.x = x  #[N,25]
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0)))

        #x_i contains the features of the target nodes for each edge
        #x_j contains the features of the source nodes for each edge
    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)     #(E, 3)
        return self.mlp_msg(msg_input)                          #(E, 64)

        #inputs = the output of message
        #index is x_i
    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=0, reduce=self.aggr, dim_size=self.x.size(0))  #(N, 64)

        #aggr_out is the output of aggregate (N,64)
        #x is original node features (N, 64)
    def update(self, aggr_out, x):
        upd_input = torch.cat([x, aggr_out], dim=1)
        
        return  self.upd_msg(upd_input) #(N, 64)


class MPNNModel(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim=64, num_layers=5): #(25,1)
        super().__init__()

        #1.
        self.lin_in = nn.Linear(in_dim, hidden_dim)
                                  #x --> (N, 64) 
                                  
        #2. layers of calling
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim = hidden_dim ,hidden_layers =32, edge_dim = edge_dim))
                                        # emb_dim = 64                              edge_dim = 1                 
          
        #3.Pooling
        self.pool = global_mean_pool #[1,64]
        
        #4. Linear
        self.lin_layer = nn.Sequential(
                                nn.Linear(hidden_dim, 1),
                                )

    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x =self.lin_in(x) #(N, 64)
        #print("BEFORE CONV: ", x.shape)
        for conv in self.convs:
    
            x += conv(x, edge_index, edge_attr)   

        x = self.pool(x, batch) 

        return self.lin_layer(x).view(-1)

model = MPNNModel(in_dim= len(dataset[0].x[0]), edge_dim=len(dataset[0].edge_attr[0]) ).to(device)
#(25,1)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
loss = nn.MSELoss()

from tqdm import tqdm
from tqdm import trange

#Creating training and testing datasets
batch_size = 32
test_size = 0.3
num_samples = len(dataset)

test_no = int(test_size * num_samples)
valid_no = test_no

train_no = len(dataset) - test_no*2

np.random.seed(42)
indices = np.arange(len(dataset))
np.random.shuffle(indices)
train_index, test_index, valid_index = indices[:train_no], indices[train_no:train_no+test_no], indices[train_no+test_no:train_no+test_no+valid_no]

# define samplers for obtaining training and testing batches
train_sampler = SubsetRandomSampler(train_index)
test_sampler = SubsetRandomSampler(test_index)
valid_sampler = SubsetRandomSampler(valid_index)

train_load = DataLoader(dataset, batch_size = batch_size, sampler = train_sampler)
test_load = DataLoader(dataset, batch_size = batch_size, sampler = test_sampler)
valid_load = DataLoader(dataset, batch_size = batch_size, sampler = valid_sampler)

#training model
def train(train_loader, model, optimizer, loss_fn):
    model.train()
    train_loss = 0 
    i = 0

    for data in tqdm(train_loader, desc="Training batches"):
        data = data.to(device)
        #print("node num:", len(data.x[:,0]))
        #print(f"MYDATA: {data}")
        y_pred = model(data)

        #if i%30 == 0:
        #    print("PRED: ",y_pred )
        #    print("LABEL: ", data.y)

        loss = loss_fn(y_pred, data.y)
        
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i+= 1
        
    train_loss /= len(train_loader)
    return train_loss


def test(test_loader, model, loss_fn):
    model.eval()
    test_loss = 0

    with torch.inference_mode():
        for data in test_loader:
            print(test_loader)
            data = data.to(device)

            test_pred = model(data)

            print("TEST_PRED:",test_pred)
            loss = loss_fn(test_pred, data.y)

            print("Data_label:", data.y)
            test_loss += loss.item()
            
    test_loss /= len(test_loader)

    return test_loss


ep = []
result1 = []
result2 = []
min = torch.inf

import numpy as np
for epoch in trange (10, desc="Processing"):

    train_loss = train(train_load, model, optimizer, loss)
    test_loss = test( test_load,model, loss)
    ep.append(epoch)
    result1.append(np.array(torch.tensor(train_loss).numpy()))
    result2.append(np.array(torch.tensor(test_loss).numpy()))

    #draw_epoch_graph(G, model_1.latest)
    print(f"Epoch {epoch} | Train Loss: {train_loss} | Test Loss: {test_loss}")

    if test_loss < min:
        min = test_loss
        torch.save(model.state_dict(), 'model_regress.pt')

print(f"Best for test Loss: {min}")
#plotting loss curves
plt.plot(ep, result1, color = "red", label = "Train_Loss")
plt.plot(ep, result2, color = "blue", label="Test_Loss")
plt.legend()
plt.title("MPNN_Regression Loss Curves")
plt.show()

model_test = MPNNModel(in_dim= len(dataset[0].x[0]), edge_dim=len(dataset[0].edge_attr[0])).to(device)
model_test.load_state_dict(torch.load('model_regress.pt', weights_only=True))
model_test.to(device)