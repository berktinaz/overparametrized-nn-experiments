import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch import optim


# Seed the random generator
np.random.seed(0)
torch.manual_seed(0)

# Hyperparameters for our network
input_dim = 3
hidden_size = 3
overparam_factors = [16,48,160,1600,16000,1]
test_data_size = int(1e4)


####### STUDENT PART #######
num_param = (input_dim + 1)*hidden_size
data_size = num_param * 4

print("Data size:", data_size)

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

V_torch = torch.from_numpy(V)
W_torch = torch.from_numpy(W)

# Set weights
with torch.no_grad():
    student[0].weight = nn.Parameter(W_torch)
    student[2].weight = nn.Parameter(V_torch)

# Generate train data
X = np.zeros((data_size,input_dim))
Y = np.zeros((data_size,1))
for i in range(data_size):
    input = np.random.normal(0,1,3)
    input = input / np.linalg.norm(input)
    X[i] = input
    Y[i] = float((student(torch.from_numpy(input)).data[0]))

X_torch = torch.from_numpy(X)
Y_torch = torch.from_numpy(Y)

# Generate test data
X_test = np.zeros((test_data_size,input_dim))
Y_test = np.zeros((test_data_size,1))
for i in range(test_data_size):
    input = np.random.normal(0,1,3)
    input = input / np.linalg.norm(input)
    X_test[i] = input
    Y_test[i] = float((student(torch.from_numpy(input)).data[0]))

X_torch_test = torch.from_numpy(X_test)
Y_torch_test = torch.from_numpy(Y_test)

for overparam in overparam_factors:
    #### TRAINING PART (TEACHER) ######
    hidden_size_s =  hidden_size * overparam
    teacher = nn.Sequential(nn.Linear(input_dim, hidden_size_s, bias=False),
                        nn.ReLU(),
                        nn.Linear(hidden_size_s, 1, bias=False))

    learning_rate = 3*1e-4
    epochs = int(1e4)

    optimizer = optim.SGD(teacher.parameters(), lr=learning_rate)

    losses = []
    losses_test = []
    for i in range(epochs):
        optimizer.zero_grad()
        y_hat = teacher(X_torch.float())
        diff = torch.abs(y_hat - Y_torch).pow(2)
        loss = torch.mean(diff) # MSE
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().numpy())
        
        if i % 100 == 0:
            teacher.eval()
            with torch.no_grad():   
                y_hat = teacher(X_torch_test.float())
                diff = torch.abs(y_hat - Y_torch_test).pow(2)
                loss_t = torch.mean(diff) # MSE
                losses_test.append(loss_t.detach().numpy())
                print(i)
            teacher.train()
    np.save("./exp1_results/hiddenSize_" + str(hidden_size) + "_overFactor_x" + str(overparam) + "_dataSize_" + str(data_size) + "_LR_" + str(learning_rate) + "_epochs_" + str(epochs) + "2.npy", losses)  
    np.save("./exp1_results/hiddenSize_" + str(hidden_size) + "_overFactor_x" + str(overparam) + "_dataSize_" + str(data_size) + "_LR_" + str(learning_rate) + "_epochs_" + str(epochs) + "test2.npy", losses_test)  
    del teacher