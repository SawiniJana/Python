import torch
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


from torchvision import datasets, transforms

import matplotlib.pyplot as plt


dataset = datasets.MNIST('./data', 
                         train = True,
                         download = True,
                         transform = transforms.ToTensor())

#Device Agnostic code
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


class MNISTPC_Dataset(data.Dataset):
    def __init__(self, 
                 dataset_path: str,
                 download: bool=False,
                 train= False,
                 ):
        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.dataset = datasets.MNIST(dataset_path,
                                         train = True,
                                         download = False,
                                         transform = self.transform
                                        )
        
        self.pc_dataset = [self.transform_2d_img_to_point_cloud(self.dataset[idx]) for idx in range(len(self.dataset))]
    
    def transform_2d_img_to_point_cloud(self, data_i:tuple) -> dict:

        img, label = data_i
        img_array = img.squeeze()
        values = img_array[img_array > 0]
        x_coord = torch.argwhere(img_array > 0.)[:,0]/img_array.shape[0]
        y_coord = torch.argwhere(img_array > 0.)[:,1]/img_array.shape[0]
        
        point_vec = torch.stack( (x_coord, y_coord, values), dim=-1)
        
        return {'point' : point_vec, 'label' : label, 'seq_length' : len(point_vec)}
        

    def __len__(self) -> int:
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return len(self.pc_dataset)
    
    def __getitem__(self, idx:int) -> dict :
        # Return the idx-th data point of the dataset
    
        return self.pc_dataset[idx]#data_point, data_label

Dataset = MNISTPC_Dataset('./data/')

class DeepSetLayer(nn.Module):
    def __init__(self, in_features:int, out_features: int, normalization:str = '', pool:str='mean') -> None:
        super().__init__()

        self.Gamma = nn.Linear(in_features, out_features)
        self.Lambda = nn.Linear(in_features,out_features)

        self.normalization = normalization
        self.pool = pool

        if normalization == 'batchnorm':
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if(self.pool == 'mean'):
            x = self.Gamma(x) + self.Lambda(x - x.mean(dim=1, keepdim=True))
        elif(self.pool == 'max'):
            x = self.Gamma(x) + self.Lambda(x - x.max(dim=1, keepdim=True))
        
        if self.normalization == "batchnorm":
            x = self.bn(x)

        return x

ds_layer = DeepSetLayer

class DeepSet(nn.Module):
    def __init__(self, in_features:int, feats:list, n_class:int, normalization:str = '', pool:str = 'mean') ->None:
        super().__init__()

        layers = []

        layers.append(DeepSetLayer(in_features = in_features, out_features = feats[0], normalization = normalization, pool = pool))
        for i in range(1, len(feats)):
            layers.append(nn.ReLU())
            layers.append(DeepSetLayer(in_features = feats[i-1], out_features = feats[i], normalization = normalization, pool = pool))

        layers.append(DeepSetLayer(in_features = feats[-1], out_features = n_class, normalization = normalization, pool = pool))
        #self.sequential = nn.Sequential(*layers)
        self.sequential = nn.ModuleList(layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #return self.sequential(x)
        for i, layer in enumerate(self.sequential):
            x = layer(x)
        
        x = x.mean(dim=1) # -- average over the points -- #
        out = F.log_softmax(x, dim=-1)
        
        return out


model = DeepSet(
    in_features = 3,
    feats = [5,10,16,16,12],
    n_class = 10
)

#loss fn and optimizer

optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

#Creating training and testing
def train(train_loader, model, optimizer, device):
    model.to(device)
    model.train()
    train_loss = 0
    
    for bacth, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        y_pred = model(data)
        loss = F.nll_loss(y_pred, target)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(train_loader)
    
    return train_loss

def test(test_loader, model, device):
    model.eval()
    test_loss = 0
    with torch.inference_mode():
        for id, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            test_pred = model(data)
            loss = F.nll_loss(test_pred, target)
            test_loss += loss.item()

        test_loss /= len(test_loader)

    return test_loss

def Create_batch(batch):
    max = 0
    for j in batch:
        if j['seq_length'] > max:
            max = j['seq_length']
    batch_points = []
    batch_label = []
    for i in batch:
        st = i['point'].numpy()
        val = i['seq_length']

        pad = max-val
        padded_str = np.zeros((pad,3))
        new_str = np.concatenate([st,padded_str],axis=0)
        batch_points.append(torch.tensor(new_str, dtype=torch.float32))
        batch_label.append(torch.tensor(i['label'], dtype = torch.long))

    batch_points = torch.stack(batch_points)
    batch_labels = torch.stack(batch_label)

    return batch_points, batch_labels



train_data =  MNISTPC_Dataset("./data", download = True, train = True)
test_data = MNISTPC_Dataset("./data", download = True, train = False)
batch_size = 32
valid_size = 0.2
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_index, valid_index = indices[split:], indices[:split]
# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_index)
valid_sampler = SubsetRandomSampler(valid_index)
train_loader = data.DataLoader(dataset=train_data, batch_size = batch_size, sampler = train_sampler, collate_fn=Create_batch)
valid_loader = data.DataLoader(train_data, batch_size = batch_size, sampler=valid_sampler,collate_fn=Create_batch)
test_loader = data.DataLoader(test_data, batch_size = batch_size, shuffle=True, collate_fn=Create_batch)

epoch = 10
test_min = np.inf
for i in range(epoch):

    result1 = train(train_loader, model, optimizer,device)
    result2 = test(test_loader, model, device)

    print(f"Train Loss: {result1} | Test Loss: {result2}")

    if result2 < test_min:
        
        torch.save(model.state_dict(), 'model.pt')
        test_min = result2

print(f"Best for: {model.state_dict()}| Test Loss: {test_min}")


def eval_model(model,valid_loader ):
    model.eval()
    true_preds = true_reds = num_preds = 0

    with torch.inference_mode():
        for data, targets in valid_loader:
            data, targets = data.to(device), targets.to(device)
            Valid_pred = model(data)
            
            pred_abels = torch.argmax(Valid_pred, dim= 1)
            
            true_reds += (pred_abels == targets).sum()

            num_preds += len(targets.squeeze())
    
    acc2 = true_reds/num_preds
    print(f"Accuracy of the model: {100.0*acc2}%")



model_test = DeepSet(in_features=3, feats=[5,10,16,16,12], n_class=10)
model_test.load_state_dict(torch.load('model.pt', weights_only=True))
model_test.to(device)
eval_model(model_test, valid_loader)







