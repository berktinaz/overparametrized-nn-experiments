import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D

# FFMPEG Directory
# plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

# Global scat
scats = []

# Variables
seed = 100
LR = 0.25
target = 1e-8
hiddenSize = 3
overFactor = 33
dataSize = hiddenSize * 16
maxFrames = 900
bound = 0.05
linewidth = 2.5
scatterSize = 25
drawLow = True # make it false to draw high scale
weight_channel = range(hiddenSize * overFactor)

# set params
if drawLow:
    scale = 0.01
    epochs = 89063
else:
    scale = 3
    epochs = 147336
    
# Load planted model
name_init = "hiddenSize_"+str(hiddenSize)+"_overFactor_x"+str(overFactor)+"_dataSize_"+str(dataSize)+"_LR_"+str(LR)
planted_name = name_init+"_target_"+str(target)+"_seed_"+str(seed)

W_planted = np.load("hiddenSize_3_seed_"+str(seed)+"_W.npy")
V_planted = np.load("hiddenSize_3_seed_"+str(seed)+"_V.npy")

# get the planted model
planted = W_planted / np.sqrt(3)

# Load low / high scale weights
name_post = "_target_"+str(target)+"_scaling_"+str(scale)+"_seed_"+str(seed)+"_weights.pt"
name = name_init + "_epochs_" + str(epochs) + name_post
weights = torch.load(name)
W = weights["W"].detach().numpy()
V = weights["V"].detach().permute(0,2,1).numpy().squeeze(-1) # iterations x channels

W_m = np.multiply(W, np.abs(np.stack([V,V,V], axis=2)))

colors = np.empty_like(V, dtype=str) # iterations x channels
colors[:] = "r"
colors[V >= 0] = "b"

# # Display the average change in weights
# print("Mean change (low scale): ", np.average(abs(W_l[-1,:,:] - W_l[0,:,:])))
# print("Mean change (hight scale): ", np.average(abs(W_h[-1,:,:] - W_h[0,:,:])))

# Start drawing the figure
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=10., azim=45)

ax.grid(False)
ax.axis("off")
ax.set_xlim(-bound, bound)
ax.set_ylim(-bound, bound)
ax.set_zlim(-bound, bound)

# draw_sphere()
#calculate vectors for "vertical" circle

# Draw the planted model
for i in range(planted.shape[0]):
    ax.plot(np.append(planted[i,0],0),np.append(planted[i,1],0),np.append(planted[i,2],0), linewidth=linewidth, color="k", alpha=0.9)

lines = [ax.plot([],[],[], "-", linewidth=linewidth, alpha=0.4)[0] for _ in weight_channel]

def draw_sphere():
    # Draw Sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

    elev = 10.0
    rot = 80.0 / 180 * np.pi
    
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='white', linewidth=0, alpha=0.2)
    a = np.array([-np.sin(elev / 180 * np.pi), 0, np.cos(elev / 180 * np.pi)])
    b = np.array([0, 1, 0])
    b = b * np.cos(rot) + np.cross(a, b) * np.sin(rot) + a * np.dot(a, b) * (1 - np.cos(rot))
    ax.plot(np.sin(u),np.cos(u),0,color='k', linestyle = 'dashed')
    horiz_front = np.linspace(0, np.pi, 100)
    ax.plot(np.sin(horiz_front),np.cos(horiz_front),0,color='k')
    vert_front = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
    ax.plot(a[0] * np.sin(u) + b[0] * np.cos(u), b[1] * np.cos(u), a[2] * np.sin(u) + b[2] * np.cos(u),color='k', linestyle = 'dashed')
    ax.plot(a[0] * np.sin(vert_front) + b[0] * np.cos(vert_front), b[1] * np.cos(vert_front), a[2] * np.sin(vert_front) + b[2] * np.cos(vert_front),color='k')

def animate(i, lines):
    if i == maxFrames - 1: # after max frames draw last frame
        i = W.shape[0] - 1
        
    global scats
    # first remove all old scatters
    for scat in scats:
        scat.remove()
    scats=[]

    if i > 0:
        for channel in weight_channel:
            lines[channel].set_data([W_m[:i+1,channel,0],W_m[:i+1,channel,1]])
            lines[channel].set_3d_properties(W_m[:i+1,channel,2])
            lines[channel].set_color(colors[i,channel])
            
    scats.append(ax.scatter( W_m[i,:,0], W_m[i,:,1], W_m[i,:,2], s=scatterSize, c=colors[i], marker="o"))
        
    print(i)
    return lines

# Animate
anim = animation.FuncAnimation(fig, animate, frames=min(maxFrames ,W.shape[0]), fargs=(lines,), blit=True)
# Save
FFwriter = animation.FFMpegWriter(fps=30, bitrate=-1, extra_args=['-vcodec', 'libx264'])
anim.save('exp3_x'+str(overFactor)+'_scale_'+str(scale)+'.mp4', writer=FFwriter)
# plt.show()