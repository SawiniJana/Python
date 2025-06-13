import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device used:{ device}")

torch.manual_seed(42)
#set some parameters
true_w = 0.3
true_b = 0.9
X = torch.linspace(0,10,100).unsqueeze(1)
Y = true_w * X + true_b

#defining the model using class
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype = torch.float, requires_grad = True))
        self.biases = nn.Parameter(torch.randn(1,  dtype = torch.float, requires_grad = True))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.biases
        
#Defining a custom dataset:
class SimpleLinearDataset(Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx], self.Y[idx]
    
# Create dataset
dataset = SimpleLinearDataset(X, Y)


#splitting into training and testing tests
train_size = int(0.8*dataset.__len__())
test_size = int(0.2*dataset.__len__())
train_set , test_set = random_split(dataset, [train_size, test_size])
#print(len(train_set[0]),len(test_set[0]))

batch_size = 20
train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(test_set)
print(len(test_loader))
torch.manual_seed(42)
model = LinearRegression()
model.to(device)

X_test = []
Y_test = []
#print(model.state_dict())
for a,b in test_loader:  
    X_test.append(a.item())
    Y_test.append(b.item())
X_test = torch.tensor(np.array(X_test))
Y_test = torch.tensor(np.array(Y_test))

loss_fn = nn.L1Loss(reduction="mean")

optimizer = torch.optim.SGD(params=model.parameters(), lr = 0.01 )

loss1= []
loss2= []
epochs = []
#TRAINING LOOP
epoch = 200

ini = 100000000000

for i in range(epoch):
    model.train()
    train_loss = 0

    for X_train, Y_train in train_loader:
        X_train, Y_train = X_train.to(device), Y_train.to(device)
        y_pred = model(X_train)
        #print(len(y_pred), len(Y_train))
        loss = loss_fn(y_pred,Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        #print(train_loss)

    train_loss = train_loss/len(train_loader)
    

        
    if i%10 == 0:
        model.eval()

        with torch.inference_mode():

            X_test,Y_test = X_test.to(device), Y_test.to(device)
            y_preds = model(X_test)
            test_loss = loss_fn(y_preds,Y_test)
            if test_loss < ini:
                ini = test_loss
                min_weights = model.weights.item()
                min_bias = model.biases.item()
                
        print(model.weights,model.state_dict())
                
            #print(f"Training Loss: {train_loss} | Test Loss: {test_loss} | epoch: {i}")

        
        loss2.append(test_loss)
        loss1.append(train_loss)
              
        epochs.append(i)


print(f"Best Test Loss: {ini}| Weights: {min_weights} | Biases: {min_bias} | model saves: {model.state_dict()}")
# Plot the loss curves
plt.plot(epochs, np.array(torch.tensor(loss1).numpy()), label="Train loss")
plt.plot(epochs, loss2, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()


from pathlib import Path

Model_path = Path("models")
Model_path.mkdir(parents=True, exist_ok = True, )

Model_name = "Excercise1.pth"
model_save_path = Model_path / Model_name
#print(f"Saving best model with weights: {best_model_state['weights']}, bias: {best_model_state['biases']}")

torch.save(obj=model.state_dict(),
           f=model_save_path)


loaded_model = LinearRegression()
loaded_model.load_state_dict(torch.load(model_save_path, weights_only=True) )
loaded_model.to(device)

loaded_model.eval()
#print(f"Loaded model: weights = {loaded_model.weights.item()}, bias = {loaded_model.biases.item()}")
with torch.inference_mode():
    preds = loaded_model(X_test)

#print(len(y_preds), len(preds))
#print(y_preds.to(device))
#print(preds.to(device))
