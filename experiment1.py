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
hidden_sizes = [3]
# hidden_sizes = [3,10,20,50,100]
# hidden_sizes = [20, 100, 200, 500]
# hidden_sizes = [50, 250, 500, 1250]
# hidden_sizes = [20]
overparam_factors = [1,2,3,6,10,100,1000]

for hidden_size in hidden_sizes:
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

    for overparam in overparam_factors:
        #### TRAINING PART (TEACHER) ######
        hidden_size_s =  hidden_size * 16 * overparam
        teacher = nn.Sequential(nn.Linear(input_dim, hidden_size_s, bias=False),
                            nn.ReLU(),
                            nn.Linear(hidden_size_s, 1, bias=False))

        learning_rate = 3*1e-4
        epochs = int(1e4)

        optimizer = optim.SGD(teacher.parameters(), lr=learning_rate)

        losses = []
        for i in range(epochs):
            optimizer.zero_grad()
            y_hat = teacher(X_torch.float())
            diff = torch.abs(y_hat - Y_torch).pow(2)
            loss = torch.mean(diff) # MSE
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().numpy())
        np.save("hiddenSize_" + str(hidden_size) + "_overFactor_x" + str(overparam) + "_dataSize_" + str(data_size) + "_LR_" + str(learning_rate) + "_epochs_" + str(epochs) + ".npy", losses)  
        del teacher
    
# TODO: For lower LR increase epoch num, Also try lower data-param ratio (1/10) and start from 200 neurons (something like that)