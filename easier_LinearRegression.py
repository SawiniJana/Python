import torch
import matplotlib.pyplot as plt
import numpy as np

#step1: Generating a random number of samples
n_sample = 1000
ratio = int(0.9 * n_sample)

#Step2: Generating random X Values
X = torch.rand((n_sample,1)) * 5000 + 500 #x_values between 500 to 5500

# Min-Max Normalization: Scale X to [0, 1]
X_max = X.max()
X_scaled = X/X_max
X = X_scaled

X_train = X[:ratio]
X_test = X[ratio:]


#Step3: Generating true parameters coz we have none
w_true = 5
b_true = 2
print("weight:", w_true,"Bias", b_true)

#Step4: Calculating Y
noise = torch.randn(n_sample, 1) * 20
Y = X*w_true + b_true + noise
Y_train = Y[:ratio]
Y_test = Y[ratio:]
print(Y)
#Step5:Initialising random parameters
w = torch.randn(1,requires_grad=True)
b = torch.randn(1,requires_grad=True)

#Defining Hyperparameters
epoch = 1000
lr = 0.1

#storing for plotting
losses = []
epo = np.arange(1000)
epo2 = []
CT = []

#Step6: Training loop
for epoch in range(epoch):
    #Forward pass
    y_pred = w*X_train + b

    #Computing Loss
    Loss = ((Y_train - y_pred)**2)/2
    Cost = torch.mean(Loss)
    losses.append(Cost.item())

    #backpropagation
    Cost.backward()


    #Updating the parameters
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        
        # Zero the gradients after updating
        w.grad.zero_()
        b.grad.zero_()


    if epoch%100 == 0:
        print(f"Training Loss:{Cost} | epoch: {epoch}")
        epo2.append(epoch)
        #Prdeicting 
        with torch.inference_mode():
            Y_pred = w*X_test+ b
            loss_test = ((Y_pred - Y_test)**2)/2
            cost_test = torch.mean(loss_test)
            CT.append(cost_test.item())
            print(f"Calculated Weight:{w} | Calculated Bias: {b} | Test Loss: {cost_test}")

plt.plot(epo,losses,label="Train loss")
plt.plot(epo2, CT, label="Test loss" )
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
