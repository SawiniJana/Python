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
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, degree, scatter, index_to_mask
import torch.nn.functional as F

from torchvision import transforms
import matplotlib.pyplot as plt

#Device Agnostic code
if torch.cuda.is_available():
    device = 'cuda'
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
        con[torch.isnan(con)] = 0.0

        return {'jet': con, 
                'sd_mass' : torch.tensor(jet_sdmass), 
                'seq_length': torch.tensor(jet_nparticles)  }

    def __len__(self):
        return self.num_entries
    
    def __getitem__(self, index):
        return self.pt_cloud(index)

jet_dataset = Jet_Dataset(file)
#print(dataset[1235])

#Create Batch
def Create_Batch(data):
    #data --> list of tensors of point clouds (32 tensors should contain size )
    list = []
    max = 0
    for tens in data:
        if tens.size(0) > max:
            max = tens.size(0)

    for tens in data:
        
        req = max - tens.size(0)
        req_zero = torch.zeros(req, tens.size(1))
        tens = torch.cat((tens, req_zero), dim=0)
        list.append(tens)
    return list

#collate_fn
def collate_fn(batch):

    jets = [item['jet'] for item in batch]
    sd_mass = torch.cat([item['sd_mass'] for item in batch], dim=0)

    padded_points = torch.stack(Create_Batch(jets), dim=0)
        # convert labels list to tensor

    return padded_points, sd_mass

jet_dataloader = data.DataLoader(dataset=jet_dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)
#a,b = next(iter(jet_dataloader))
#print(a.shape,b.shape)

#Creating Deepset layer
class DeepSetLayer(nn.Module):
    def __init__(self, in_features, out_features, normalisation, pool='mean'):
        super().__init__()

        self.gamma = nn.Linear(in_features, out_features)
        self.Lambda = nn.Linear(in_features, out_features)
        self.pool = pool

        self.normalisation = normalisation
        if normalisation == "batchnorm":
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        if self.pool == "mean":
            out = self.gamma(x) + self.Lambda(x - x.mean(dim=1, keepdim=True))

        elif self.pool == "max":
            out = self.gamma(x) + self.Lambda(x - x.mean(dim=1, keepdim=True))

        if self.normalisation == "batchnorm":
            #print(out.shape)
            out = out.transpose(1,2)
            out = self.bn(out)
            out = out.transpose(1,2)

        return out

#Creating the module
class DeepSetModule(nn.Module):
    def __init__(self, feats ,norm,pool, num_layers = 4):
        super().__init__()

        #Layers of calling
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(DeepSetLayer(feats[i], feats[i+1], norm, pool))
            self.convs.append(nn.ReLU())


        self.lin_layer = nn.Sequential(
                                nn.Linear(feats[i+1], 1),
                                nn.ReLU())
        
    def forward(self, x): #(B,N, feats)
        batch = torch.tensor(x.size(0))

        for conv in self.convs:
            x = conv(x) #(B,N,16)

        x = x.mean(dim=1) #(B,16)

        return self.lin_layer(x).view(-1) #(B)


model = DeepSetModule([23,15,30,16,10],"batchnorm","mean").to(device)
#print(model(pt_dict["points"]))
optimizer = torch.optim.SGD(model.parameters(),lr = 1e-4)
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
train_load = data.DataLoader(jet_dataset, batch_size = batch_size, sampler = train_sampler, collate_fn=collate_fn)
test_load = data.DataLoader(jet_dataset, batch_size = batch_size, sampler = test_sampler, collate_fn=collate_fn)
valid_loader = data.DataLoader(jet_dataset, batch_size = 1, sampler = valid_sampler, collate_fn=collate_fn)

#training model
def train(train_loader, model, optimizer, loss_fn):
    model.train()
    train_loss = 0

    for data, label in tqdm(train_loader, desc="Training batches"):
        data = data.to(device)
        #print(f"MYDATA: {data}")
        y_pred = model(data)
        #print(y_pred.dtype, label.dtype)
        
        loss = loss_fn(y_pred, label)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(train_loader)
    return train_loss

def test(test_loader, model, loss_fn):
    model.eval()
    test_loss = 0
    i = 0

    with torch.inference_mode():
        for data, label in test_loader:

            data = data.to(device)

            test_pred = model(data)
            
            #print("TEST_PRED:",test_pred)
            loss = loss_fn(test_pred, label)
            #print("Data_label:", data.label )
            test_loss += loss.item()

            if i == 30:
                print("My_pred: ", test_pred)
                print("Test_pred: ", label)

            i+= 1

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
        torch.save(model.state_dict(), 'model.pt')

print(f"Best for test Loss: {min}")
#plotting loss curves
plt.plot(ep, result1, color = "red", label = "Train_Loss")
plt.plot(ep, result2, color = "blue", label="Test_Loss")
plt.legend()
plt.title("REGRESSION_DEEPSETS_Loss Curves")
plt.show()

#VALIDATING
import random
idx = random.randint(0, len(jet_dataset))
valid = jet_dataset[idx]
for dat,label in valid_loader:
    dat, label = dat.to(device), label.to(device)
    break
model_test = DeepSetModule([23,15,30,16,10],"batchnorm","mean").to(device)
model_test.load_state_dict(torch.load('model.pt', weights_only=True))
model_test.to(device)

valid_pred = model_test(dat)
print("PREDICTED SD_MASS: ", valid_pred)
print("TRUTH SD_MASS: ", label)