import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

# Define a custom dataset
import numpy as np
import pandas as pd


class EEEC_Dataset(Dataset):
    def __init__(self, csv_file):
        data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
        self.z1 = data[:, 0]
        self.z2 = data[:, 1]
        self.z3 = data[:, 2]
        self.Etilde = data[:, 3]
        self.mt = data[:, 4]

    def __len__(self):
        return len(self.z1)

    def __getitem__(self, idx):
        x = np.array([self.z1[idx], self.z2[idx], self.z3[idx], self.mt[idx]])
        w = self.Etilde[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(w, dtype=torch.float32)


def compute_minmax(dataset):
    zs1, zs2, zs3 = [], [], []
    for x, _ in dataset:
        zs1.append(x[0].item())
        zs2.append(x[1].item())
        zs3.append(x[2].item())
    return (
        min(zs1), max(zs1),
        min(zs2), max(zs2),
        min(zs3), max(zs3)
    )


class Preprocessor:
    def __init__(self, z1min, z1max, z2min, z2max, z3min, z3max):
        self.z1min, self.z2min, self.z3min = z1min, z2min, z3min
        self.z1max, self.z2max, self.z3max = z1max, z2max, z3max

    def transform_z1(self, z1):
        return np.log10(z1 / self.z1min) / np.log10(self.z1max / self.z1min)

    def transform_z2(self, z2):
        return np.log10(z2 / self.z2min) / np.log10(self.z2max / self.z2min)

    def transform_z3(self, z3):
        return np.log10(z3 / self.z3min) / np.log10(self.z3max / self.z3min)

    def transform_mt(self, mt):
        return (mt - 170) / (180 - 170)

    def transform(self, X):
        z1 = X[:, 0]
        z2 = X[:, 1]
        z3 = X[:, 2]
        mt = X[:, 3]
        return np.stack(
            [
                np.log10(z1 / self.z1min) / np.log10(self.z1max / self.z1min),
                np.log10(z2 / self.z2min) / np.log10(self.z2max / self.z2min),
                np.log10(z3 / self.z3min) / np.log10(self.z3max / self.z3min),
                (mt - 170) / (180 - 170),
            ],
            axis=1,
        )


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


class EEEC_Model(nn.Module):
    def __init__(self, in_shape: int, hidden_units: int, out_shape: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_shape, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, out_shape),
        )

    def forward(self, x: torch.Tensor):
        return torch.clamp(self.layers(x).squeeze(), -20, 20)


dataset = EEEC_Dataset("/home/sawini-jana/Documents/EEEC_observables_172.5.csv")

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

torch.manual_seed(42)
training_set, testing_set = random_split(dataset, [train_size, test_size])

z1min, z1max, z2min, z2max, z3min, z3max = compute_minmax(training_set)

preprocessor = Preprocessor(
    z1min, z1max,
    z2min, z2max,
    z3min, z3max,
)

train_processed = TransformedDataset(training_set, preprocessor)
test_processed = TransformedDataset(testing_set, preprocessor)

train_loader = DataLoader(train_processed, batch_size=128, shuffle=True)
test_loader = DataLoader(test_processed, batch_size=128, shuffle=False)

model_test = EEEC_Model(in_shape=4, hidden_units=256, out_shape=1)
model_test.load_state_dict(
    torch.load("/home/sawini-jana/Documents/model.pt", weights_only=True)
)

csv_file = f"/home/sawini-jana/Documents/EEEC_observables_172.5.csv"

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"File not found: {csv_file}")

# HERE WE COMPARE FOR ONE mt EEECNN vs EEECMC

#Figure 2

# for z = z1 + z2 + z3
# za <= 0.02
# EQUILATERAL TRIANGLE

zp = df.z1 + df.z2 + df.z3
mask_eq = (((df.z3 - df.z1)) <= 0.02)

zp = zp[mask_eq]
bins = np.linspace(0.0, 0.1 , 50)
centers = 0.5 * (bins[:-1] + bins[1:])
bin_widths = bins[1:] - bins[:-1]

# EEEC MC
w = df.weight[mask_eq]
eeec_mc = np.zeros(len(centers))
for i in range(len(centers)):
    m_val2 = (zp >= bins[i]) & (zp < bins[i+1])
    eeec_mc[i] = w[m_val2].sum()

eeec_mc /= np.trapezoid(eeec_mc, centers)
# EEEC NN
X = np.stack([df.z1, df.z2, df.z3, df.mt], axis=1)
X = X[mask_eq]
X = torch.tensor(preprocessor.transform(X), dtype=torch.float32)

with torch.no_grad():
    eeec_nn = torch.exp(model_test(X)).numpy()

eeec_nn_binned = np.zeros(len(centers))
for i in range(len(centers)):
    m_val = (zp >= bins[i]) & (zp < bins[i+1])
    eeec_nn_binned[i] = eeec_nn[m_val].sum()

eeec_nn_binned /= np.trapezoid(eeec_nn_binned, centers)

# plotting
plt.plot(centers,eeec_mc,"o", label=f"MC 172.5 GeV")
plt.plot(centers, eeec_nn_binned,"-", label=f"NN 172.5 GeV")
plt.xlabel(r"$\zeta$")
plt.ylabel("Normalized EEEC")
plt.title("Equilateral triangles")
plt.tight_layout()
plt.legend()
plt.savefig("/home/sawini-jana/Documents/FiG2.png")
plt.close()


# FIGURE 4a
z3 = df.z3.values
w = df.weight.values

bins = np.linspace(z3.min(), z3.max(), 50)
centers = 0.5 * (bins[:-1] + bins[1:])

# EEEC MC
eeec_mc = np.zeros(len(centers))
for i in range(len(centers)):
    m_val2 = (z3 >= bins[i]) & (z3 < bins[i + 1])
    eeec_mc[i] = w[m_val2].sum()

eeec_mc /= np.trapezoid(eeec_mc, centers)

# EEEC NN
X = np.stack([df.z1, df.z2, df.z3, df.mt], axis=1)
X = torch.tensor(preprocessor.transform(X), dtype=torch.float32)

with torch.no_grad():
    eeec_nn = torch.exp(model_test(X)).numpy()

eeec_nn_binned = np.zeros(len(centers))
for i in range(len(centers)):
    m_val = (z3 >= bins[i]) & (z3 < bins[i + 1])
    eeec_nn_binned[i] = eeec_nn[m_val].sum()

eeec_nn_binned /= np.trapezoid(eeec_nn_binned, centers)

# plotting
plt.plot(centers, eeec_mc, "o", label=f"MC 172.5 GeV")
plt.plot(centers, eeec_nn_binned, "-", label=f"NN 172.5 GeV")
plt.xlabel(r"$\zeta_{\max}$")
plt.ylabel("Normalized EEEC")
plt.title("EEEC as function of zmax = z3")
plt.legend()
plt.savefig("/home/sawini-jana/Documents/FiG4a.png")
plt.close()
