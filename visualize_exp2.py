import matplotlib.pyplot as plt
import numpy as np

train_target = 1e-6
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
scales = [0.01,1,2,3,5] # 1e-8 target v3 - post overparametrization fix
losses = np.array([1.0535, 85.087, 200, 300, 700])*1e-6 # 1e-8 target v3 - post overparametrization fix
plt.plot(scales, losses, marker="o")
plt.plot(scales,[1e-8]*5, marker="o")
plt.xscale('log')
plt.yscale("log")
plt.xlabel("scale")
plt.ylabel("MSE")
plt.legend(["test error","training error"])
ax = plt.gca()
ax.set_aspect('auto')
ax.set_xlim(1e-2, 10)
ax.set_ylim(1e-9, 1e-2)
print(ax.get_ylim())
plt.savefig("exp2_1e-8v3.png")