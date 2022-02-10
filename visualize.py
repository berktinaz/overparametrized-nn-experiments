import matplotlib.pyplot as plt
import numpy as np

hidden_sizes = [3]
# hidden_sizes = [20,100,200,500]
# hidden_sizes = [50,250,500,1250]
# overparam_factors = [1,16,48,160,1600,16000]
overparam_factors = [1,10]
learning_rate = 0.3333333333333333
epochs = int(2e3)
input_dim = 3

# test_x = np.arange(0, 2e3, 100)

for hidden_size in hidden_sizes:
    plt.figure()
    for overparam in overparam_factors:
        num_param = (input_dim + 1)*hidden_size
        data_size = num_param * 4
        losses = np.load("./exp1_results/hiddenSize_" + str(hidden_size) + "_overFactor_x" + str(overparam) + "_dataSize_" + str(data_size) + "_LR_" + str(learning_rate) + "_epochs_" + str(epochs) + ".npy")
        plt.plot(list(range(1,epochs+1)), np.sqrt(losses), linewidth=4) #for train
        # plt.plot(test_x, np.sqrt(losses)) #for test
        plt.yscale('log')
        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_ylim([1e-3, 1e0])
        plt.xlabel("Iterations", fontsize=16, labelpad=0)
        plt.ylabel(r'$\sqrt{MSE}$', fontsize=16, labelpad=-7)
        # plt.title("#Neuron:" + str(hidden_size) + ", LR=" + str(round(learning_rate,5))+ ", Train Errors")
    # plt.legend(["x1","x16","x48","x160","x1600","x16000"])
    # plt.legend(["GD"], fontsize=14)
    plt.legend(["GD","GD with extra hidden units"], fontsize=14)
    plt.savefig("./exp1_results/numNeuron_" + str(hidden_size) + "_lr_" + str(round(learning_rate,5)) +"_train.png")
    
    