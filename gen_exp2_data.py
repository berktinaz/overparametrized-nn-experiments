import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch import optim


# Seed the random generator
np.random.seed(1)
torch.manual_seed(1)

# Hyperparameters for our network
input_dim = 3
hidden_size = 200
data_size = int(1e5)

# Build the student network
student = nn.Sequential(nn.Linear(input_dim, hidden_size, bias=False),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1, bias=False))
print(student)

# Generate weights
W = np.zeros((hidden_size,input_dim))
V = np.zeros((1,hidden_size))
for i in range(hidden_size):
    input = np.random.normal(0,1,input_dim)
    input = input / np.linalg.norm(input)
    W[i] = input
    V[0,i] = np.random.choice([-1,1], 1, p=[0.5,0.5]) / np.sqrt(hidden_size)
        
# print(V)
# print(np.sum(V))
#print(np.linalg.norm(student[0].weight[0].detach().numpy()))

V_torch = torch.from_numpy(V)
W_torch = torch.from_numpy(W)

# Set weights
with torch.no_grad():
    student[0].weight = nn.Parameter(W_torch)
    student[2].weight = nn.Parameter(V_torch)

# print(student[2].weight) 
# print(student[0].weight)     

# Generate data
X = np.zeros((data_size,input_dim))
Y = np.zeros((data_size,1))
for i in range(data_size):
    input = np.random.normal(0,1,3)
    input = input / np.linalg.norm(input)
    X[i] = input
    Y[i] = float((student(torch.from_numpy(input)).data[0]))

X_torch = torch.from_numpy(X)
Y_torch = torch.from_numpy(Y)
print(X.shape, Y.shape)

data = {}
data["X"] = X_torch
data["Y"] = Y_torch
np.save("exp2_data_200.npy", data)
