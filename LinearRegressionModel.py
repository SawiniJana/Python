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



class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype = torch.float))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
    

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

batch_size = 20 #how much we want
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset)
dataiter = iter(test_loader)
data = next(dataiter)
X_test,y_test = data

model_1 = LinearRegression()
model_1.to(device)

def loss_fn(y_hat, y):
    l = (y_hat - y).abs()
    return l.mean()

optimizer = torch.optim.SGD(params=model_1.parameters(), lr = 0.09)

epochs = 500
epoch_count = []
loss_val = []
test_val = []


for epoch in range(epochs):

    model_1.train()
    train_loss = 0
    
    for X_train, y_train in train_loader:
        X_train, y_train = X_train.to(device), y_train.to(device)
        y_pred = model_1(X_train)
        loss = loss_fn(y_hat = y_pred, y = y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(model_1.bias)
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model_1.eval()
    with torch.inference_mode():
        test_loss = 0
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model_1(X_batch)
            loss = loss_fn(y_pred, y_batch)
            test_loss += loss.item()
        test_loss /= len(test_loader)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_val.append(train_loss)
        test_val.append(test_loss)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

print("Params:", model_1.state_dict())
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


plot_predictions(predictions=y_pred.detach().numpy())

# Plot the loss curves
plt.plot(epoch_count, np.array(torch.tensor(loss_val).numpy()), label="Train loss")
plt.plot(epoch_count, test_val, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()
