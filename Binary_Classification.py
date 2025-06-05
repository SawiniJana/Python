import torch
import numpy as np
import matplotlib.pyplot as plt
import math

#Generating no. of samples
n_sample = 1000

#Let there be clusters of points
CL1 = torch.rand((n_sample//2,2)).float() * 5.0 + 2
CL2 = torch.rand((n_sample//2,2)).float() * 5.0 + 6
X = torch.cat([CL1,CL2],dim = 0)
print(X)
#Compute Y
Y = torch.cat([torch.zeros(n_sample//2,1), torch.ones(n_sample//2,1)])
#print(Y)

print(X[:,1].shape)
print(Y.shape)
ratio = int(0.9*n_sample)

X_train = X[:ratio]
X_test = X[ratio:]
Y_train = Y[:ratio]
Y_test = Y[ratio:]

#Define the model parameters
W = torch.randn((2,1), requires_grad=True)  
b = torch.randn(1, requires_grad=True)  

# Step 3: Define Hyperparameters
lr = 0.1
epochs = 1000

#storing for plotting
losses = []
epo = np.arange(1000)
epo2 = []
CT = []

def sigmoid(x):
    return 1/(1 + torch.exp(-x))
def binary_cross_entropy(y_pred,y_train):
    return -torch.mean(y_train * torch.log(y_pred)  + (1 - y_train) * torch.log(1 - y_pred))

#Step4: Training loop
for epoch in range(epochs):
    #Forward pass
    Z = X_train @ W + b
    Y_pred = sigmoid(Z)
    #Computing Loss
    Cost = binary_cross_entropy(Y_pred,Y_train)
    losses.append(Cost.item())

    #backpropagation
    Cost.backward()


    #Updating the parameters
    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad
        
        # Zero the gradients after updating
        W.grad.zero_()
        b.grad.zero_()

    if epoch%100 == 0:
        print(f"Training Loss:{Cost} | epoch: {epoch}")
        epo2.append(epoch)
        #Prdeicting 
        with torch.inference_mode():
            Y_pred = sigmoid(X_test @ W + b) 
            cost_test = binary_cross_entropy(Y_pred, Y_test)
            
            CT.append(cost_test.item())
            print(f"Calculated Weight:{W} | Calculated Bias: {b} | Test Loss: {cost_test}")

#print(epo,epo2)
plt.plot(epo,losses,label="Train loss")
plt.plot(epo2, CT, label="Test loss" )
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

#Making Prediction
x = int(input("Enter a co-ordinate within 10: "))
y = int(input("Enter a co-ordinate within 10: "))
X_TEST = torch.tensor((x,y))
Y_TEST = sigmoid(X_TEST[0]* W[0] + X_TEST[1]*W[1] + b)
if Y_TEST > 0.5:
    print("It is closer to 6,6")
else:
    print("It is closer to 2,2")
