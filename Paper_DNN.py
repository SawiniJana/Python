import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

#Define a custom dataset
import numpy as np

class EEEC_Dataset(Dataset):
    def __init__(self, csv_file):
        data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
        #z1,z2,z3,E',mt
        self.z1 = data[:,0]
        self.z2 = data[:,1]
        self.z3 = data[:,2]
        self.Etilde = data[:,3]
        self.mt = data[:,4]

    def __len__(self):
        return len(self.z1)

    def __getitem__(self, idx):
        x = np.array([self.z1[idx], self.z2[idx], self.z3[idx], self.mt[idx]])
        w = self.Etilde[idx] 
        return torch.tensor(x, dtype=torch.float32), torch.tensor(w, dtype=torch.float32)

#Create dataset
from torch.utils.data import random_split

dataset = EEEC_Dataset("/home/sawini-jana/hep/EEEC_1000samples.csv")

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

training_set, testing_set = random_split(dataset, [train_size, test_size])
#print(training_set[:][0][0],training_set[:][1][0])

#Computing the minimum and maximum values required for normalisation
def compute_minmax(dataset):
    zs1, zs2, zs3 = [], [], []

    for x, _ in dataset:
        zs1.append(x[0].item())
        zs2.append(x[1].item())
        zs3.append(x[2].item())

    return (
        min(zs1), max(zs1),min(zs2), max(zs2),min(zs3), max(zs3))
#Computing only for the training samples and using the normalising values to test 
z1min, z1max, z2min, z2max, z3min, z3max = compute_minmax(training_set)

class Preprocessor:
    def __init__(self, z1min, z1max,z2min, z2max,z3min, z3max):
        self.z1min,self.z2min,self.z3min = z1min,z2min,z3min
        self.z1max,self.z2max,self.z3max = z1max,z2max,z3max

    def transform_z1(self, z1):
        x1 = np.log10(z1 / self.z1min) / np.log10(self.z1max / self.z1min)
        return max(0.0, min(1.0, x1)) #sensitive to top mass only, improving training
    def transform_z2(self, z2):
        x2 = np.log10(z2 / self.z2min) / np.log10(self.z2max / self.z2min)
        return  max(0.0, min(1.0, x2))
    def transform_z3(self, z3):
        x3 =  np.log10(z3 / self.z3min) / np.log10(self.z3max / self.z3min)
        return max(0.0, min(1.0, x3))
    def transform_mt(self, mt):
        return (mt - 170)/(180-170) #max mt taken is 180.0 GeV

    def transform(self, x):
        z1, z2, z3, mt = x
        return np.array([
            self.transform_z1(z1),self.transform_z2(z2),
            self.transform_z3(z3),self.transform_mt(mt)]
            )
#Applying the normalisation
class TransformedDataset(Dataset):
    def __init__(self, base_dataset, preprocessor):
        self.base_dataset = base_dataset
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, w = self.base_dataset[idx]
        x = self.preprocessor.transform(x.numpy())
        return torch.tensor(x, dtype=torch.float32), w

preprocessor = Preprocessor(
    z1min, z1max,
    z2min, z2max,
    z3min, z3max,
)
#Processing the training and testing sets
train_processed = TransformedDataset(training_set, preprocessor)
test_processed = TransformedDataset(testing_set, preprocessor)

#Storing them in DataLoader 
from torch.utils.data import DataLoader

train_loader = DataLoader(train_processed, batch_size=128, shuffle=True)
test_loader = DataLoader(test_processed, batch_size=128, shuffle=False)

#Neural Network Model
class EEEC_Model(nn.Module):
    def __init__(self, in_shape: int, hidden_units: int, out_shape: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_shape, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, out_shape),
            ) 
        #Here the neural network is learning log p 
        #where p = probability density of observing an EEEC triplet with angles (z1,z2,z3)
        #batch size * 1
     
    def forward(self, x:torch.Tensor):
        #print("Model: ",self.layers(x))
        return self.layers(x) #clamping down on extreme values

torch.manual_seed(42)
model_0 = EEEC_Model(in_shape=4, hidden_units=256, out_shape=1)
#batch_size * out_shape
#here we output = ln(EEEC)

#loss and optimizer
def loss_fn(log_eeec_pred, weights, lambda_norm=1.0):
    #convergence of loss function based on KL divergence between EEEC data and EEEC calc
    #weights: Etilde, shape Batch_size * 1

    eeec_pred = torch.exp(log_eeec_pred)
    #we get EEEC observables (probability density of observing an EEEC triplet)

    # First term: weighted negative log-likelihood
    term1 = -torch.mean(weights * log_eeec_pred)

    # Second term: normalization regularization
    term2 = torch.log(torch.mean(eeec_pred))
    #print("Loss: ", term1 + lambda_norm * term2)
    return term1 + lambda_norm * term2

optimizer = torch.optim.Adam(params = model_0.parameters(), lr=0.00008, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,   # drop every 10 epochs
    gamma=0.5       # multiply LR by 0.5
)

def train(model, train_loader, optimizer, loss_fn):
    train_loss = 0

    for batch, (X_train, Y_train) in enumerate(train_loader):
        model.train()

        EEEC_Data = model(X_train)
        #print(EEEC_Data.shape)
        loss = loss_fn(EEEC_Data, Y_train)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    train_loss /= len(train_loader)

    return train_loss

def test(model, val_loader, loss_fn):
    test_loss = 0
    model.eval()

    with torch.inference_mode():
        for batch, (X_test, y_test) in enumerate(val_loader):
            test_pred = model(X_test)
            test_loss += loss_fn(test_pred, y_test)
         
        test_loss /= len(test_loader)
    
    return test_loss
    
epochs = 30
ep = []
result1 = []
result2 = []
test_min = torch.inf

for i in range(epochs):
    train_loss = train(model_0, train_loader, optimizer, loss_fn)
    test_loss = test(model_0, test_loader, loss_fn)

    ep.append(i)
    result1.append(train_loss.item())
    result2.append(test_loss.item())    
    if i%20 == 0:
        print(f"Epoch {i} | Train Loss: {train_loss} Test loss: {test_loss}")
    if test_loss < test_min:
        test_min = test_loss
        torch.save(model_0.state_dict(), '/home/sawini-jana/Documents/model.pt')


#Plot the loss curves
plt.plot(ep, result1, label="Train loss")
plt.plot(ep, result2, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.savefig("/home/sawini-jana/Documents/DNN_Loss_Curves_1000.png")


