import torch
import numpy as np
import matplotlib.pyplot as plt

#Generating no. of samples
n_sample = 1000

#Feature1:
X_1 = torch.rand((n_sample, 1)) * 5000 + 500
X_max = X_1.max()
X_scaled_1 = X_1/X_max
X_1 = X_scaled_1

#Feature2:
X_2 = torch.randint(1,6,(n_sample,1 )).float()
#Feature3:
X_3 = torch.rand((n_sample,1))*50
X_max = X_3.max()
X_scaled_3 = X_3/X_max
X_3 = X_scaled_3

#Combine all features into one
X = torch.cat((X_1,X_2,X_3),dim=1)
ratio = int(0.9*n_sample)
X_train = X[:ratio]
X_test = X[ratio:]

#Generate parameters
true_weights = torch.tensor([0.2, 30.0, -4.0])  
true_intercept = 50 

#Compute Y
noise = torch.randn(n_sample, 1) * 20
Y = X[:, 0] * true_weights[0] + X[:, 1] * true_weights[1] + X[:, 2] * true_weights[2] + true_intercept + true_intercept + noise
#print(Y)
Y_train = Y[:ratio]
Y_test = Y[ratio:]

#Define the model parameters
W = torch.randn(3, requires_grad=True)  
b = torch.randn(1, requires_grad=True)  

# Step 3: Define Hyperparameters
lr = 0.1
epochs = 1000

#storing for plotting
losses = []
epo = np.arange(1000)
epo2 = []
CT = []


#Step4: Training loop
for epoch in range(epochs):
    #Forward pass
    Y_pred = (
        X_train[:,0]*W[0] +
        X_train[:,1]*W[1] + 
        X_train[:,2]*W[2] + 
        b
    )
    #Computing Loss
    Loss = ((Y_train - Y_pred)**2)/2
    Cost = torch.mean(Loss)
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
            Y_pred = W[0]*X_test[:,0] + W[1]*X_test[:,1] + W[2]*X_test[:,2] + b
            loss_test = ((Y_pred - Y_test)**2)/2
            cost_test = torch.mean(loss_test)
            CT.append(cost_test.item())
            print(f"Calculated Weight:{W} | Calculated Bias: {b} | Test Loss: {cost_test}")

#print(epo,epo2)
plt.plot(epo,losses,label="Train loss")
plt.plot(epo2, CT, label="Test loss" )
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

