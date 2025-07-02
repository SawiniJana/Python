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

        #Labels
        jet_class = -1

        if (self.branch["label_QCD"].to_numpy()[idx:idx+1]==1) : jet_class = 0

        if ((self.branch["label_Tbqq"].to_numpy()[idx:idx+1] == 1) or
              (self.branch["label_Tbl"].to_numpy()[idx:idx+1]==1)) : jet_class = 1
        
        if ((self.branch["label_Zqq"].to_numpy()[idx:idx+1] == True) or
            (self.branch["label_Wqq"].to_numpy()[idx:idx+1] == True)) : jet_class = 0
        
        if ((self.branch["label_Hbb"].to_numpy()[idx:idx+1] == True) or
            (self.branch["label_Hcc"].to_numpy()[idx:idx+1] == True) or
            (self.branch["label_Hgg"].to_numpy()[idx:idx+1] == True) or
            (self.branch["label_H4q"].to_numpy()[idx:idx+1] == True) or
            (self.branch["label_Hqql"].to_numpy()[idx:idx+1] == True)) : jet_class = 2
        
        #part
        i_feat = [torch.tensor(ak.flatten(self.branch[i][idx:idx+1].to_numpy())).unsqueeze(0) for i in self.part]
        part_feat = torch.cat((i_feat), dim=0).T #shape(39, 16)

        #concantenate part and jet together (x)
        con = torch.concatenate((torch.tensor(jet_feat),part_feat), dim = 1)   
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
        data.label = torch.tensor(jet_class, dtype=torch.long)
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
#print(dataset[1235])


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

model = GCNModule(in_dim= len(jet_dataset[0].x[0]),hidden_dim = 64, out_dim = 3)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay=5e-4)
loss = nn.CrossEntropyLoss()

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

#training model
def train(train_loader, model, optimizer, loss_fn):
    model.train()
    train_loss = 0

    for data in tqdm(train_loader, desc="Training batches"):
        data = data.to(device)
        #print(f"MYDATA: {data}")
        y_pred = model(data.x, data.edge_index, data.batch)
        #print(y_pred.dtype, label.dtype)
        
        loss = loss_fn(y_pred, data.label)
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

            data = data.to(device)

            test_pred = model(data.x, data.edge_index, data.batch)
            
            #print("TEST_PRED:",test_pred)
            loss = loss_fn(test_pred, data.label)
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
plt.title("GNN Loss Curves")
plt.show()

def eval_model(model,valid_loader ):
    model.eval()
    true_preds = num_preds = 0

    with torch.inference_mode():
        for data in valid_loader:
            data = data.to(device)
            Valid_pred = model(data.x, data.edge_index, data.batch)
            
            pred_labels = torch.argmax(Valid_pred, dim= 1)
            
            true_preds += (pred_labels == data.label).sum()

            num_preds += len(data.label)
    
    acc2 = true_preds/num_preds
    print(f"Accuracy of the model: {100.0*acc2}%")

model_test = GCNModule(in_dim= len(jet_dataset[0].x[0]),hidden_dim = 64, out_dim = 3)
model_test.load_state_dict(torch.load('model.pt', weights_only=True))
model_test.to(device)
eval_model(model_test, valid_loader)



