import matplotlib.pyplot as plt
import numpy as np

hidden_size = 20
overparam = 3
learning_rate = 1e-2
epochs = 342810
input_dim = 3
fact = 4
target_err = 1e-07
scaling = 1

plt.figure()
data_size = int(np.floor(input_dim * hidden_size / fact))
losses = np.load("hiddenSize_" + str(hidden_size) + "_overFactor_x" + str(overparam) + "_dataSize_" + str(data_size) + "_LR_" + str(learning_rate) + "_epochs_" + str(epochs) + "_target_" + str(target_err) + "_scaling_" + str(scaling) + ".npy")  
print(losses)
plt.plot(list(range(1,epochs+1)), losses)
plt.yscale('log')
ax = plt.gca()
ax.set_ylim([1e-8, 1e1])
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.title("#Neuron:" + str(hidden_size) + ", LR=" + str(learning_rate) + ", Fact=" + str(fact))
plt.savefig("numNeuron_" + str(hidden_size) + "_lr_" + str(learning_rate) + "_fact_" + str(fact) + ".png")

