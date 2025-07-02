import torch.utils.data as data
from torch_geometric.loader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import uproot
import numpy as np
import torch_geometric

from torch_geometric.data import Data
import torch
import awkward as ak
import numpy

import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree, scatter, index_to_mask
import torch.nn.functional as F

from torchvision import transforms
import matplotlib.pyplot as plt

#Device Agnostic code
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

file = "C:\\Users\\SAWINI JANA\\Downloads\\JetClass_example_100k.root"

class Jet_Dataset(data.Dataset):

    def __init__(self, filepath, k = 5):
        super().__init__()
        self.file = uproot.open(filepath)
        self.tree = self.file["tree"] #awkward high-level object
        self.branch = self.tree.arrays()
        self.num_entries = len(self.branch)
        
        self.jet = self.tree.keys(filter_name="jet_*")
        self.part = self.tree.keys(filter_name="part_*")

        self.k = k

    def pt_cloud(self, idx):
         #jet, label, part, aux
         #JET
        jet_pt = self.branch["jet_pt"].to_numpy()[idx :idx+1]   #extracting jet_py from all jet entries and then extracting a particular idx
        jet_eta = self.branch["jet_eta"].to_numpy()[idx : idx+1]
        jet_phi = self.branch["jet_phi"].to_numpy()[idx : idx+1]
        jet_energy = self.branch["jet_energy"].to_numpy()[idx : idx+1]
        jet_nparticles = self.branch["jet_nparticles"].to_numpy()[idx : idx+1]
        jet_sdmass = self.branch["jet_sdmass"].to_numpy()[idx : idx+1]
        jet_tau21 = self.branch['jet_tau2'].to_numpy()[idx:idx+1]/self.branch['jet_tau1'].to_numpy()[idx:idx+1] #what is this???
        jet_tau32 = self.branch['jet_tau3'].to_numpy()[idx:idx+1]/self.branch['jet_tau2'].to_numpy()[idx:idx+1]
        jet_tau43 = self.branch['jet_tau4'].to_numpy()[idx:idx+1]/self.branch['jet_tau3'].to_numpy()[idx:idx+1]

        jet_feat = np.stack([jet_pt, jet_eta, jet_phi, jet_energy, jet_tau21, jet_tau32, jet_tau43], axis=1)
        #print(jet_feat.shape)
        jet_feat = np.stack([jet_feat]*int(jet_nparticles.item()), axis = 0).squeeze() #shape (39, )
        #print(torch.tensor(jet_feat).shape)
        
        #part
        i_feat = [torch.tensor(ak.flatten(self.branch[i][idx:idx+1].to_numpy())).unsqueeze(0) for i in self.part]
        part_feat = torch.cat((i_feat), dim=0).T #shape(39, 16)

        #concantenate part and jet together (x)
        con = torch.concatenate((torch.tensor(jet_feat),part_feat), dim = 1)   
        #con = (con - con.mean(dim=0)) / (con.std(dim=0) + 1e-8) --> NORMALIZE
        con[torch.isnan(con)] = 0.0

        #edge_index
        part_eta_ak = self.branch["part_deta"] #awkward array for entire dataset
        part_eta_ak = part_eta_ak[idx:idx+1] #awkward array for 1 entry (39 particles)
        part_eta_flat  = ak.flatten(part_eta_ak) #to turn the awkward nested structure to 1D array
        part_eta = part_eta_flat.to_numpy()

        part_phi = ak.flatten(self.branch["part_dphi"][idx:idx+1]).to_numpy()
        eta_phi_pos = torch.stack([torch.tensor(part_eta), torch.tensor(part_phi)], dim=-1)
        edge_index = torch_geometric.nn.pool.knn_graph(x = eta_phi_pos, k = self.k)

        #edge_attributes
        src, target = edge_index
        part_del_eta = part_eta[src] - part_eta[target] #since src is index of eta nodes and target of phi nodes
        part_del_phi = part_phi[src] - part_phi[target]
        part_del_R = torch.hypot(torch.from_numpy(part_del_eta),torch.from_numpy(part_del_phi)).view(-1,1)

        #data
        data = Data(x = con, edge_index= edge_index, edge_deltaR = part_del_R)

        data.sd_mass = torch.tensor(jet_sdmass)
        data.seq_length = torch.tensor(jet_nparticles) 

        return data

    def __len__(self):
        return self.num_entries
    
    def __getitem__(self, index):
        return self.pt_cloud(index)

#In general, Embedding dimension is the number of features each node has after embedding
#or Message Passing .. of the shape [num_nodes, emb_dim]
#predict sd_mass as well as class labels 

#Edge Dimension is the number of features each edge carries, 
#in the shape of edge_attr... [num_edges, edge_dim]

jet_dataset = Jet_Dataset(file)
#print(jet_dataset[1235])

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
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

        #x_i contains the features of the target nodes for each edge
        #x_j contains the features of the source nodes for each edge
    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)     #(E, 3)
        return self.mlp_msg(msg_input)                          #(E, 64)

        #inputs = the output of message
        #index is x_i
    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=0, reduce=self.aggr)  #(N, 64)

        #aggr_out is the output of aggregate (N,64)
        #x is original node features (N, 64)
    def update(self, aggr_out, x):
        upd_input = torch.cat([x, aggr_out], dim=1)
        #print(upd_input.shape)
        return self.upd_msg(upd_input) #(N, 64)

class MPNNModel(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim=64, num_layers=3): #(25,1)
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
                                nn.ReLU())

    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_deltaR, data.batch
        x =self.lin_in(x) #(N, 64)
        #print("BEFORE CONV: ", x.shape)
        for conv in self.convs:
    
            x += conv(x, edge_index, edge_attr)   


        #print("AFTER CONV: ",x.shape)
        x = self.pool(x, batch) 
        #print("AFTER POOL", x.shape)
        return self.lin_layer(x).view(-1)

model = MPNNModel(in_dim= len(jet_dataset[0].x[0]), edge_dim=len(jet_dataset[0].edge_deltaR[0]) ).to(device)
#(25,1)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay=5e-4)
loss = nn.MSELoss()

from tqdm import tqdm
from tqdm import trange

#Creating training and testing datasets
batch_size = 32
test_size = 0.2
indices = list(range(1650))
np.random.shuffle(indices)
split = int(np.floor(test_size * 1650))
test_index, valid_index, train_index,   = indices[:split], indices[split:split*2], indices[split*2:]


# define samplers for obtaining training and testing batches
train_sampler = SubsetRandomSampler(train_index)
test_sampler = SubsetRandomSampler(test_index)
valid_sampler = SubsetRandomSampler(valid_index)
train_load = DataLoader(jet_dataset, batch_size = batch_size, sampler = train_sampler)
test_load = DataLoader(jet_dataset, batch_size = batch_size, sampler = test_sampler)
valid_loader = DataLoader(jet_dataset, batch_size = batch_size, sampler = valid_sampler)

all_masses = []
for data in train_load:
    all_masses.append(data.sd_mass)
all_masses = torch.cat(all_masses, dim=0)
sd_mass_mean = all_masses.mean().item()
sd_mass_std = all_masses.std().item()

#training model
def train(train_loader, model, optimizer, loss_fn):
    model.train()
    train_loss = 0 
    i = 0

    for data in tqdm(train_loader, desc="Training batches"):
        data = data.to(device)
        #print(f"MYDATA: {data}")
        y_pred = model(data)

        if i%30 == 0:
            print("PRED: ",y_pred )
            print("LABEL: ", data.sd_mass)

        normalized_target = (data.sd_mass - sd_mass_mean) / sd_mass_std
        loss = loss_fn(y_pred, normalized_target)
        
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

            data = data.to(device)

            test_pred = model(data)

            #print("TEST_PRED:",test_pred)
            normalized_target = (data.sd_mass - sd_mass_mean) / sd_mass_std
            loss = loss_fn(test_pred, normalized_target)

            #print("Data_label:", data.label )
            test_loss += loss.item()
            
    test_loss /= len(test_loader)

    return test_loss


ep = []
result1 = []
result2 = []
min = torch.inf

import numpy as np
for epoch in trange (5, desc="Processing"):
    train_loss = train(train_load, model, optimizer, loss)
    test_loss = test( test_load,model, loss)
    ep.append(epoch)
    result1.append(np.array(torch.tensor(train_loss).numpy()))
    result2.append(np.array(torch.tensor(test_loss).numpy()))

    #draw_epoch_graph(G, model_1.latest)
    print(f"Epoch {epoch} | Train Loss: {train_loss} | Test Loss: {test_loss}")

    if test_loss < min:
        min = test_loss
        torch.save(model.state_dict(), 'model.pt')

print(f"Best for test Loss: {min}")
#plotting loss curves
plt.plot(ep, result1, color = "red", label = "Train_Loss")
plt.plot(ep, result2, color = "blue", label="Test_Loss")
plt.legend()
plt.title("MPNN_Regression Loss Curves")
plt.show()

#VALIDATING
import random
idx = random.randint(0, len(jet_dataset))
valid = jet_dataset[idx]
data = valid.to(device)

model_test = MPNNModel(in_dim= len(jet_dataset[0].x[0]), edge_dim=len(jet_dataset[0].edge_deltaR[0]))
model_test.load_state_dict(torch.load('model.pt', weights_only=True))
model_test.to(device)

valid_pred = model_test(data)
pred_mass = valid_pred * sd_mass_std + sd_mass_mean
print("PREDICTED SD_MASS: ", pred_mass)
print("TRUTH SD_MASS: ", data.sd_mass)