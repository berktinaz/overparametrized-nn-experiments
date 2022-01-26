import matplotlib.pyplot as plt
import numpy as np

hidden_sizes = [3]
# hidden_sizes = [20,100,200,500]
# hidden_sizes = [50,250,500,1250]
overparam_factors = [1,2,3,6,10,100,1000]
learning_rate = 0.00030000000000000003
epochs = int(1e4)
input_dim = 3
fact = 4

for hidden_size in hidden_sizes:
    plt.figure()
    for overparam in overparam_factors:
        num_param = (input_dim + 1)*hidden_size
        data_size = num_param * 4
        num_param_s = data_size * 4 * overparam
        losses = np.load("hiddenSize_" + str(hidden_size) + "_overFactor_x" + str(overparam) + "_dataSize_" + str(data_size) + "_LR_" + str(learning_rate) + "_epochs_" + str(epochs) + ".npy")
        plt.plot(list(range(1,epochs+1)), np.sqrt(losses))
        plt.yscale('log')
        ax = plt.gca()
        ax.set_ylim([1e-3, 1e0])
        plt.xlabel("Iterations")
        plt.ylabel("sqrt(error)")
        plt.title("#Neuron:" + str(hidden_size) + ", LR=" + str(learning_rate) + ", Fact=" + str(fact))
    plt.legend(["x1","x2","x3","x6","x10","x100","x1000"])
    plt.savefig("numNeuron_" + str(hidden_size) + "_lr_" + str(learning_rate) + "_fact_" + str(fact) + ".png")
    
    