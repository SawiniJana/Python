# Import PyTorch
import torch
from torch import nn

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt

#Creating training and testing data
train_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download=True,
    transform = ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

#img, label = train_data[0]
#print(len(train_data.data), train_data.data[0].shape)
class_names = train_data.classes

#plotting 
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze().numpy(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)
plt.show()

#creating batches
from torch.utils.data import DataLoader

train_loader = DataLoader(train_data,
                          batch_size = 32,
                          shuffle = True
                          )

test_loader = DataLoader(test_data,
                         batch_size = 32,
                         shuffle = False
                         )

train_data_1, train_label_1 = next(iter(train_loader))
#print(train_data_1.shape) #1X28X28 of 32 samples 


#in the flattening step, the input will be ...
flatten_model  = nn.Flatten()
x = train_data_1[0]
output = flatten_model(x)
#print(len(output.squeeze()))

#creating module
from torch import nn
class FashionMNIST(nn.Module):
    def __init__(self,
                 in_shape: int,
                 hidden_units: int,
                 out_shape: int
                 ):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels= in_shape,
                out_channels= hidden_units,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels= hidden_units,
                out_channels= hidden_units,
                kernel_size = 3,
                stride = 1,
                padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2)  
            )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding= 1),
            nn.ReLU())
         
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features= hidden_units*14*14, out_features= out_shape )
        )
    

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.layer_stack(x)
        return x
    
    
torch.manual_seed(42)
model_0 = FashionMNIST(in_shape= 1, 
                       hidden_units=10,
                       out_shape=len(class_names)
                       )

#setting up loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr = 0.1
                            )

def acc_fn(Y1, Y2):
    correct = torch.eq(Y1,Y2).sum().item()
    acc = (correct / len(Y1)) * 100
    return acc

epochs = 3

for epoch in range(epochs):
    train_loss = train_acc = 0
    #Training
    print("HI1")
    for batch, (X_train,Y_train) in enumerate(train_loader):
        model_0.train()

        y_pred = model_0(X_train)
        loss = loss_fn(y_pred, Y_train)
        train_loss += loss  
        #no of iterations = no. of batches
        
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
    print("HI2")
    train_loss /= len(train_loader)
    train_acc += acc_fn(Y_train,y_pred.argmax(dim=1))
    train_acc /= len(train_loader)
    test_loss = test_acc = 0
    
    model_0.eval()
    with torch.inference_mode():
        for X_test, y_test in test_loader:
            test_pred = model_0(X_test)
            test_loss += loss_fn(test_pred, y_test)
            test_acc += acc_fn(y_test, test_pred.argmax(dim=1) )
            #print("HI3")
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
    print(f"Train Loss: {train_loss}| Test loss: {test_loss} | Train ACC: {train_acc} | Test ACC: {test_acc}")       


#MAKing PRediction
import random
random.seed(42)
test_samples = []
test_labels = []

id = random.randint(0,32)
sample, label = next(iter(test_loader))
sample = sample[id]
label = label[id]

plt.imshow(sample.squeeze(), cmap="gray") 
plt.title(label.item())
plt.show()

#Making prediction
with torch.inference_mode():
    sample = torch.unsqueeze(sample, dim = 0)
    pred_logit  = model_0(sample)
    pred_prob = torch.softmax(pred_logit.squeeze(dim = 0), dim = 0)
    pred_classes = pred_prob.argmax(dim=0)
    print(pred_classes)

plt.imshow(sample.squeeze(), cmap="gray")
plt.title(f"Truth: {class_names[label.item()]}  | Predicted: {class_names[pred_classes.item()]}")
plt.show()