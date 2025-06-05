import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device used: {device}")

# Set a seed for reproducibility
torch.manual_seed(42)

# Generate synthetic data: y = 3x + 2 + noise
true_w = 3
true_b = 2
X = torch.linspace(0, 10, 100).unsqueeze(1)  # Shape: (100, 1)
y = true_w * X + true_b + torch.randn_like(X)  # Add noise

# Define a custom dataset
class SimpleLinearDataset(Dataset):
    def __init__(self, X, y):
        #we can also np.loadtxt and load data from another file here
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataset
dataset = SimpleLinearDataset(X, y)

# Split into training and testing sets (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
batch_size = 20 #how much we want
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

dataiter1 = iter(train_loader)
dataiter2 = iter(test_loader)
data1 = next(dataiter1)
data2 = next(dataiter2)
X_train,y_train = data1
X_test, y_test = data2

#plotting predictions
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):

  plt.figure(figsize=(10, 7))
  plt.scatter(train_data.numpy(), train_labels.numpy(), c="b", s=4, label="Training data")
  plt.scatter(test_data.numpy(), test_labels.numpy(), c="g", s=4, label="Testing data")
  if predictions is not None:
    plt.scatter(test_data.numpy(), predictions, c="r", s=4, label="Predictions")
  plt.show()


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,out_features=1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
    
model_1 = LinearRegression()
model_1.to(device)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr = 0.01)

epochs = 200

X_train = X_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
y_train = y_train.to(device)

epoch_count = []
loss_val = []
test_val = []

for epoch in range(epochs):

    model_1.train()
    y_pred = model_1(X_train)
    loss = loss_fn(y_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model_1.eval() #train() makes it in training mode while eval() on evaluation mode
    with torch.inference_mode():
        text_pred = model_1(X_test) #on testing set
        test_loss = loss_fn(text_pred, y_test) #loss based on testing set
    
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_val.append(loss)
        test_val.append(test_loss)
        print(f"Epoch: {epoch} | Loss: {loss } | Test loss: {test_loss}")


#plot_predictions(predictions=y_pred.detach().numpy())

# Plot the loss curves
plt.plot(epoch_count, np.array(torch.tensor(loss_val).numpy()), label="Train loss")
plt.plot(epoch_count, test_val, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()