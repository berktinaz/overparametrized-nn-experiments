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
input_dim = 3
hidden_size = 3
overparam = 33

num_param = (input_dim + 1)*hidden_size
train_data_size = num_param * 4
test_data_size = int(1e5)
print("Data size:", train_data_size)

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

np.save("hiddenSize_3_seed_100_W.npy", W)
np.save("hiddenSize_3_seed_100_V.npy", V)

V_torch = torch.from_numpy(V)
W_torch = torch.from_numpy(W)

# Set weights
with torch.no_grad():
    student[0].weight = nn.Parameter(W_torch)
    student[2].weight = nn.Parameter(V_torch)

# print(student[2].weight) 
# print(student[0].weight)     

# Generate test data
X = np.zeros((test_data_size,input_dim))
Y = np.zeros((test_data_size,1))
for i in range(test_data_size):
    input = np.random.normal(0,1,3)
    input = input / np.linalg.norm(input)
    X[i] = input
    Y[i] = float((student(torch.from_numpy(input)).data[0]))

X_torch_test = torch.from_numpy(X)
Y_torch_test = torch.from_numpy(Y)
print(X_torch_test[-1], Y_torch_test[-1])

# Generate training data
X = np.zeros((train_data_size,input_dim))
Y = np.zeros((train_data_size,1))
for i in range(train_data_size):
    input = np.random.normal(0,1,3)
    input = input / np.linalg.norm(input)
    X[i] = input
    Y[i] = float((student(torch.from_numpy(input)).data[0]))

X_torch = torch.from_numpy(X)
Y_torch = torch.from_numpy(Y)
print(X.shape, Y.shape)
print(X_torch[-1], Y_torch[-1])
# # Load test data
# test_data = np.load("./exp2_results/exp2_data.npy", allow_pickle=True)
# test_data = test_data.item()
# print(test_data["X"])

##### Training Part ######
scalings = [0.01, 3]
# scalings = [0.01]
hidden_size_s =  hidden_size * overparam
teacher_o = nn.Sequential(nn.Linear(input_dim, hidden_size_s, bias=False),
                                nn.ReLU(),
                                nn.Linear(hidden_size_s, 1, bias=False))
print(teacher_o)

## takes in a module and applies the specified weight initialization
def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
    # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))

# teacher.apply(weights_init_normal)
torch.nn.init.xavier_normal_(teacher_o[0].weight)
torch.nn.init.xavier_normal_(teacher_o[2].weight)

for scaling in scalings:
    teacher = copy.deepcopy(teacher_o)
    #scale weighting
    with torch.no_grad():
        teacher[0].weight *= scaling
        teacher[2].weight *= scaling
    
    all_W = copy.deepcopy(teacher[0].weight.detach().unsqueeze(0))
    all_V = copy.deepcopy(teacher[2].weight.detach().unsqueeze(0))
    
    print(all_W)

    learning_rate = 0.25
    target_err = 1e-8
    epoch = 0

    optimizer = optim.SGD(teacher.parameters(), lr=learning_rate)

    loss = 1
    losses = []
    with open("hiddenSize_" + str(hidden_size) + "_overFactor_x" + str(overparam) + "_dataSize_" + str(train_data_size) + "_LR_" + str(learning_rate) + "_target_" + str(target_err) + "_scaling_" + str(scaling) + "_seed_" + str(seed) + ".txt", 'w') as f:
        while loss > target_err:
            optimizer.zero_grad()
            y_hat = teacher(X_torch.float())
            # print(y_hat)
            diff = torch.abs(y_hat - Y_torch).pow(2)
            loss = torch.mean(diff) # MSE
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().numpy())
            epoch += 1
            if epoch <= 2000:
                all_W = torch.cat((all_W,teacher[0].weight.detach().unsqueeze(0)),0)
                all_V = torch.cat((all_V,teacher[2].weight.detach().unsqueeze(0)),0)
            elif epoch % 1000 == 0:
                all_W = torch.cat((all_W,teacher[0].weight.detach().unsqueeze(0)),0)
                all_V = torch.cat((all_V,teacher[2].weight.detach().unsqueeze(0)),0)
            if epoch % 1000 == 0:
                print(epoch, loss)
                f.write(str(epoch) + ": " + str(loss))
                f.write('\n')
        
        all_weights = {"W": all_W, "V": all_V}
        save_name = "hiddenSize_" + str(hidden_size) + "_overFactor_x" + str(overparam) + "_dataSize_" + str(train_data_size) + "_LR_" + str(learning_rate) + "_epochs_" + str(epoch) + "_target_" + str(target_err) + "_scaling_" + str(scaling) + "_seed_" + str(seed)   
        np.save(save_name + ".npy", losses)  
        torch.save(all_weights, save_name + "_weights.pt")
        # Save model
        # 5torch.save(teacher.state_dict(), "target_" + str(target_err) + "scale_" + str(scaling) + ".pt")
        
        # Test the network
        teacher.eval()
        with torch.no_grad():  
            y_hat = teacher(X_torch_test.float())
            diff = torch.abs(y_hat - Y_torch_test).pow(2)
            loss = torch.mean(diff) # MSE
            f.write("Scaling: " + str(scaling) + "Loss: " + str(loss))
            print("Scaling: ", scaling, "Loss: ", loss)
        # print(teacher[0].weight)
        
        del teacher