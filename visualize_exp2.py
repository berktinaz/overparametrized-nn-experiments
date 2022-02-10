import matplotlib.pyplot as plt
import numpy as np

train_target = 1e-8
# scales = [0.01,0.05,0.1,0.5,1,2,3,5] #1e-6 target v1
# losses = np.array([33,53,44,35,43,121,564,1442])*1e-4 # 1e-6 target v1
# scales = [0.05,0.1,0.5,1,2,3,5] # 1e-5 target v2
# losses = np.array([220,244,96,65,71,1265,13176])*1e-4 # 1e-5 target v2
# scales = [0.01,0.1,1,2,3,5] # 1e-5 target v1
# losses = np.array([9,20,17,160,1040,1667])*1e-4 # 1e-5 target v1
# scales = [0.01,0.05,0.1,0.5,1,2,3,5] # 1e-5 target v3 with seed 10
# losses = np.array([25,21,29,67,18,92,224,2817])*1e-4 # 1e-5 target v3 with seed 10
# scales = [0.01,0.05,0.1,0.5,1,2,3,5] # 1e-5 target v4 with seed 10 + teacher is fixed in the beginning
# losses = np.array([27,26,35,31,26,74,311,2817])*1e-4 # 1e-5 target v4 with seed 10 + teacher is fixed in the beginning
# scales = [0.05,0.1,0.5,1,2,3,5] # 1e-5 target v5 + teacher is fixed in the beginning
# losses = np.array([31,30,47,22,49,308,2614])*1e-4 # 1e-5 target v5 + teacher is fixed in the beginning
# scales = [0.05,0.1,0.5,1,2,3,5] # 1e-6 target v2 + teacher is fixed in the beginning
# losses = np.array([8,13,15,30,78,139,611])*1e-4 # 1e-6 target v2 + teacher is fixed in the beginning
# scales = [0.01,0.05,0.1,0.5,1,2,3,5] # 1e-8 target v1 teacher is fixed in the beginning
# losses = np.array([57,63,83,46,85,344,1591,12744])*1e-4 # 1e-8 target v1 teacher is fixed in the beginning
# scales = [0.01,0.05,0.1,0.5,1,2,3,5] # 1e-8 target v2 teacher is fixed in the beginning
# losses = np.array([207,210,194,48,65,332,1503,11110])*1e-4 # 1e-8 target v2 teacher is fixed in the beginning
# scales = [0.01,1,2,3,5] # 1e-8 target v3 - post overparametrization fix
# losses = np.array([1.0535, 85.087, 200, 300, 700])*1e-6 # 1e-8 target v3 - post overparametrization fix
# scales = [0.01,0.03,0.1] # 1e-8 target v4 - post overparametrization fix, no seed
# losses = np.array([4.5710e-07,1.0131e-06,1.5491e-06]) # 1e-8 target v4 - post overparametrization fix, no seed
# scales = [0.01,0.03,0.1,0.3,1,3,10] # 1e-7 target v1 - post overparametrization fix, no seed, overparam 100
# losses = np.array([2.9937e-06, 3.1005e-06,3.4133e-06,1.1495e-05,9.2672e-05,0.0006,0.0584]) # 1e-7 target v1 - post overparametrization fix, no seed, overparam 100
# scales = [0.01,0.03,0.1,0.32,1,3,6,9,10,11,12,13,14,15,16,17,17.5] # 1e-8 target v5 - post overparametrization fix, seed 100
# losses = np.array([1.5932e-05,1.9082e-05,4.5967e-05,5.8333e-05,0.0002,0.0006,0.0066,0.0324,0.0492,0.0720,0.1019,0.1407,0.1196,0.0841,0.0184,0.0138,0.0038]) # 1e-8 target v5 - post overparametrization fix, seed 100
# scales = [1,5,10,11,12,13,14,15,16,17,18,18.25] # right side test
# losses = [0.0001,0.0013,0.0211,0.0312,0.0444,0.0578,0.0403,0.0345,0.0369,0.0472,0.0540,0.2776] #right side test
scales = [0.01,0.03,0.1,0.32,1,3,10,14] # 1e-8 target v5 - post overparametrization fix, seed 100, subset
losses = np.array([1.5932e-05,1.9082e-05,4.5967e-05,5.8333e-05,0.0002,0.0006,0.0492,0.1196]) # 1e-8 target v5 - post overparametrization fix, seed 100, subset
# scales = [0.001,0.003,0.01,0.03,0.1,0.3,1,3] #1e-8 target x4, lr 0.33
# losses = [4.8006e-08,1.9594e-05,3.4882e-05,3.9007e-05,3.2687e-05,1.4598e-05,8.9069e-06,1.9840e-05]
plt.plot(scales,[1e-8]*len(losses), color="tab:orange",marker="o",markersize=10, linewidth=4)
# plt.plot(scales, losses, color="tab:blue", marker="o",markersize=10, linewidth=4)
plt.xscale('log')
plt.yscale("log")
plt.xlabel("Scale", fontsize=16)
plt.ylabel("MSE", fontsize=16,labelpad=0)
plt.legend(["training error"], loc="upper left", fontsize=14)#,"test error"
ax = plt.gca()
ax.set_aspect('auto')
ax.set_xlim(1e-2, 15)
ax.set_ylim(1e-9, 2e-1)
ax.tick_params(axis='x',labelsize=12)
ax.tick_params(axis='y',labelsize=12)
print(ax.get_ylim())
plt.savefig("exp2_1e-8v5_simp.png")