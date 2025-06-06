import torch
import matplotlib.pyplot as plt
import numpy as np

n_samples = 1000
#Custer1
X1 = torch.rand((n_samples,1))*2 - 1  #radius
Y1 = (1 - X1**2)**(0.5) + torch.rand((n_samples,1))*0.2 - 0.1
Y1[:n_samples//2] = - Y1[:n_samples//2]

#Cluster2
X2 = torch.rand((n_samples,1))*4 - 2
Y2 = (4 - X2**2)**(0.5) + torch.rand((n_samples,1))*0.2 - 0.1
Y2[:n_samples//2] = - Y2[:n_samples//2]

CL1 = torch.cat((X1,Y1), dim =1)
CL2 = torch.cat((X2,Y2), dim = 1)
X = torch.cat((CL1,CL2) ,dim = 0)
Y = torch.cat([torch.zeros(n_samples,1), torch.ones(n_samples,1)])

#Converting to X-train, Y-train, X-test, Y-test
perm = torch.randperm(X.size(0))  # random permutation of indices
X = X[perm]
Y = Y[perm]
ratio = int(0.9 * X.size(0))
X_train, Y_train = X[:ratio], Y[:ratio]
X_test, Y_test = X[ratio:],Y[ratio:]

plt.scatter(X[:,0], X[:,1], color="red")
plt.show()

from torch import nn
class CircleModelV(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features = 2, out_features = 10)
        self.layer_2 = nn.Linear(in_features = 10, out_features = 10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.Relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.Relu(self.layer_2(self.Relu(self.layer_1(x)))))

model = CircleModelV()

#Loss and optimizer
#def loss_fn(y_pred,y_train):

#    y_pred = 1/(1 + torch.exp(-y_pred))
#    print(y_pred)
#    return -torch.mean(y_train * torch.log(y_pred)  + (1 - y_train) * torch.log(1 - y_pred))
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


#Fit the model
torch.manual_seed(42)
epochs = 1000

#Storing values
epoch_count = []
test_val = []
loss_val = []


for epoch in range(epochs):
    #Forward Pass
    y = model(X_train.squeeze())
    
    #Calculate loss 
    loss = loss_fn(y, Y_train)
    
    #Optimizer zero grad
    optimizer.zero_grad()

    #Loss backward
    loss.backward()

    #Optimizer step
    optimizer.step()

    #Testing
    model.eval()
    with torch.inference_mode():
        #Forward pass
        y_val = model(X_test.squeeze())
        
        #Loss
        test_loss = loss_fn(y_val,Y_test)

    if epoch% 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss} ")
    
        epoch_count.append(epoch)
        test_val.append(test_loss)
        loss_val.append(loss)

plt.plot(epoch_count, np.array(torch.tensor(loss_val).numpy()), label="Train loss")
plt.plot(epoch_count, test_val, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

x = float(input("Enter a co-ordinate of 1 or 2: "))
y = float(input("Enter a co-ordinate of 1 or 2: "))
X_TEST = torch.tensor((x,y))
Y_TEST = model(X_TEST.squeeze())
if torch.round(torch.sigmoid(Y_TEST)) == 0:
    print("It is of radii 1")
else:
    print("It is of radii 2")
