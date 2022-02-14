from random import uniform
import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
from torch import nn
from torch import optim

# Seed the random generator
seed = 100
np.random.seed(seed)
torch.manual_seed(seed)
  
# Hyperparameters for our network
input_dim = 2
hidden_size = 3
overparam = 50
learning_rate = 0.1
target_err = 1e-8
scalings = [3, 0.01]
set_bias = True

num_param = (input_dim + 1)*hidden_size
train_data_size = num_param * 2
test_data_size = int(1e5)
print("Data size:", train_data_size)

save_name = "./exp4_results/inputDim_" + str(input_dim) + "_hiddenSize_" + str(hidden_size) + "_overFactor_x" + str(overparam) + "_dataSize_" + str(train_data_size) + "_LR_" + str(learning_rate) + "_target_" + str(target_err)

# Build the student network
student = nn.Sequential(nn.Linear(input_dim, hidden_size, bias=set_bias),
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
    
if set_bias:
    bias = np.random.normal(0,1,hidden_size)
    print("bias is: ", bias)

# Save initial weights
np.save("./exp4_results/inputDim_" + str(input_dim) + "_hiddenSize_" + str(hidden_size) + "_seed_" + str(seed) + "_W.npy", W)
np.save("./exp4_results/inputDim_" + str(input_dim) + "_hiddenSize_" + str(hidden_size) + "_seed_" + str(seed) + "_V.npy", V)
if set_bias:
    np.save("./exp4_results/inputDim_" + str(input_dim) + "_hiddenSize_" + str(hidden_size) + "_seed_" + str(seed) + "_bias.npy", bias) 

# Set weights
V_torch = torch.from_numpy(V)
W_torch = torch.from_numpy(W)
if set_bias:
    bias_torch = torch.from_numpy(bias)
with torch.no_grad():
    student[0].weight = nn.Parameter(W_torch)
    student[2].weight = nn.Parameter(V_torch)
    if set_bias:
        student[0].bias = nn.Parameter(bias_torch)

# Generate test data
X = np.zeros((test_data_size,input_dim))
Y = np.zeros((test_data_size,1))
for i in range(test_data_size):
    input = np.random.normal(0,1,input_dim) 
    input = input / np.linalg.norm(input)
    X[i] = input
    Y[i] = float((student(torch.from_numpy(input)).data[0]))

X_torch_test = torch.from_numpy(X)
Y_torch_test = torch.from_numpy(Y)

# Generate training data
X = np.zeros((train_data_size,input_dim))
Y = np.zeros((train_data_size,1))
for i in range(train_data_size):
    input = np.random.normal(0,1,input_dim) 
    input = input / np.linalg.norm(input)
    X[i] = input
    Y[i] = float((student(torch.from_numpy(input)).data[0]))
    
np.save("./exp4_results/inputDim_" + str(input_dim) + "_hiddenSize_" + str(hidden_size) + "_seed_" + str(seed) + "_bias_" + str(set_bias) + "_train_X.npy", X)
np.save("./exp4_results/inputDim_" + str(input_dim) + "_hiddenSize_" + str(hidden_size) + "_seed_" + str(seed) + "_bias_" + str(set_bias) + "_train_Y.npy", Y)

X_torch = torch.from_numpy(X)
Y_torch = torch.from_numpy(Y)
print(X.shape, Y.shape)
print(X_torch[-1], Y_torch[-1])

##### Training Part ######
hidden_size_s =  hidden_size * overparam
teacher_o = nn.Sequential(nn.Linear(input_dim, hidden_size_s, bias=set_bias),
                                nn.ReLU(),
                                nn.Linear(hidden_size_s, 1, bias=False))
print(teacher_o)

# Initialize teacher with Xavier Normal
torch.nn.init.xavier_normal_(teacher_o[0].weight)
# torch.nn.init.xavier_normal_(teacher_o[0].bias)
torch.nn.init.xavier_normal_(teacher_o[2].weight)

for scaling in scalings:
    epoch = 0
    loss = 1
    losses = []
    
    # Copy original teacher
    teacher = copy.deepcopy(teacher_o)
    
    # Scale the weights
    with torch.no_grad():
        teacher[0].weight *= scaling
        teacher[2].weight *= scaling
        if set_bias:
            teacher[0].bias *= scaling
            print(teacher[0].bias)
    
    optimizer = optim.SGD(teacher.parameters(), lr=learning_rate)

    with open( save_name + "_scaling_" + str(scaling) + "_seed_" + str(seed) + "_bias_" + str(set_bias) + ".txt", 'w') as f:
        while loss > target_err:
            optimizer.zero_grad()
            y_hat = teacher(X_torch.float())
            diff = torch.abs(y_hat - Y_torch).pow(2)
            loss = torch.mean(diff) # MSE
            loss.backward()
            optimizer.step()
            epoch += 1
            if epoch % 1000 == 0:
                print(epoch, loss)
                f.write(str(epoch) + ": " + str(loss))
                f.write('\n')
        
        final_weights = {"W": teacher[0].weight.detach(), "V": teacher[2].weight.detach()}
        torch.save(final_weights, save_name + "_epochs_" + str(epoch) + "_scaling_" + str(scaling) + "_seed_" + str(seed) + "_bias_" + str(set_bias) +"_weights.pt")

        # Test the network
        teacher.eval()
        with torch.no_grad():  
            y_hat = teacher(X_torch_test.float())
            diff = torch.abs(y_hat - Y_torch_test).pow(2)
            loss = torch.mean(diff) # MSE
            f.write("Scaling: " + str(scaling) + "Loss: " + str(loss))
            print("Scaling: ", scaling, "Loss: ", loss)

        del teacher