import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from torch import nn
from matplotlib import cm
import matplotlib.colors as colors

# Variables
seed = 100
LR = 0.1
target = 1e-8
inputDim = 2
hiddenSize = 3
overFactor = 18
dataSize = hiddenSize * (inputDim + 1) * 2
linewidth = 2.5
scatterSize = 10
hidden_size_s =  hiddenSize * overFactor
set_bias = True
plot_area = (-10,10)
view_angle = 180

# Seed
np.random.seed(seed)

# Load training data
name_init = "./exp4_results/inputDim_" + str(inputDim) + "_hiddenSize_"+str(hiddenSize)+"_overFactor_x"+str(overFactor)+"_dataSize_"+str(dataSize)+"_LR_" + str(LR) + "_target_"+str(target)
X = np.load("./exp4_results/inputDim_" + str(inputDim) + "_hiddenSize_" + str(hiddenSize) + "_seed_" + str(seed) + "_bias_" + str(set_bias) + "_train_X.npy")
Y = np.load("./exp4_results/inputDim_" + str(inputDim) + "_hiddenSize_" + str(hiddenSize) + "_seed_" + str(seed) + "_bias_" + str(set_bias) + "_train_Y.npy")

# Create Grid
X_1 = np.expand_dims(np.linspace(plot_area[0],plot_area[1],100), axis=1)
X1_V, X2_V = np.meshgrid(X_1, X_1)
inp_t = np.column_stack((X1_V.flatten(),X2_V.flatten()))
inp = np.vstack((X,inp_t))
inp = inp_t
print(inp)

# Create model from the weights
model = nn.Sequential(nn.Linear(inputDim, hidden_size_s, bias=False),
                            nn.ReLU(),
                            nn.Linear(hidden_size_s, 1, bias=False))
    
def get_model_predictions(scale, epochs, input):
    # Load final weights
    name_post = "_scaling_"+str(scale)+"_seed_"+str(seed)+"_bias_" + str(set_bias) +"_weights.pt"
    name = name_init + "_epochs_" + str(epochs) + name_post
    weights = torch.load(name)
    W = weights["W"]
    V = weights["V"]

    # Evaluate points
    with torch.no_grad():
        model[0].weight = nn.Parameter(W)
        model[2].weight = nn.Parameter(V)
        model.eval()
        Y_hat = model(torch.from_numpy(input).float()).detach().numpy()
    
    return Y_hat

#### MAIN #######  
scale_low = 0.01
scale_high = 3
# epochs_low = 155105
epochs_high = 495030
# Y_hat_low = get_model_predictions(scale=scale_low, epochs=epochs_low, input=inp)
Y_hat_high = get_model_predictions(scale=scale_high, epochs=epochs_high, input=inp)

# x1 = np.random.uniform(1,4,50)
# x2 = np.random.uniform(1,4,50)
# unif_low = get_model_predictions(scale=scale_low, epochs=epochs_low, input=np.column_stack((x1,x2)))
# unif_high = get_model_predictions(scale=scale_high, epochs=epochs_high, input=np.column_stack((x1,x2)))

# Start printing
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=20., azim=view_angle)
# plt.scatter( W_m[:,0], W_m[:,1], s=scatterSize, c=colors, marker="o")
# plt.show()
# exit()
# surf = ax.plot_trisurf(inp[:,0], inp[:,1], Y_hat_high.squeeze() - Y_hat_low.squeeze(), norm=colors.Normalize(vmin=-0.05, vmax=0.05),cmap=cm.jet, alpha=0.5, shade=True)
# surf = ax.plot_trisurf(inp[:,0], inp[:,1], Y_hat_low.squeeze(), norm=colors.Normalize(vmin=-0.05, vmax=0.05),cmap=cm.jet, alpha=0.5, shade=True)
surf = ax.plot_surface(X1_V, X2_V, np.reshape(Y_hat_high, (100,100)), cmap=cm.jet, alpha=0.9, shade=True) # norm=colors.Normalize(vmin=-0.5, vmax=0.5),
# ax.plot_trisurf(inp[:,0], inp[:,1], Y_hat_low.squeeze(), color="r", alpha=0.5, shade=True)
# ax.plot_trisurf(inp[:,0], inp[:,1], Y_hat_high.squeeze(), color="b", alpha=0.5, shade=True)
fig.colorbar(surf, shrink=0.5, aspect=5, pad = 0.1)
# ax.scatter(X[:,0],X[:,1], Y, color="r", s=scatterSize, alpha=1)
# ax.scatter(x1, x2, unif_high, color="k", s=scatterSize, alpha=0.8)
plt.savefig('./exp4_results/exp4_x'+str(overFactor)+'_scale_'+str(scale_high)+'_seed_'+str(seed) + "_bias_" + str(set_bias) +'.png', dpi=500)
