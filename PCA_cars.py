import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
data = pd.read_csv("Cars93.csv")

Horsepower = (data['Horsepower'] - data['Horsepower'].mean())/data['Horsepower'].std()
Length = (data['Length'] - data['Length'].mean())/data['Length'].std()
Width = (data['Width'] - data['Width'].mean())/data['Width'].std()
FTC = (data['Fuel.tank.capacity'] - data['Fuel.tank.capacity'].mean())/data['Fuel.tank.capacity'].std()

M = np.vstack((Horsepower, Length, Width, FTC))

cov = np.cov(M)

val, vec = np.linalg.eig(cov)

pro1 = vec[:,0].dot(M)
pro2 = vec[:,1].dot(M)

#%%

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, label='1')

ax.scatter(pro1, pro2, alpha=0)
ax.set_xlabel("First Principal component")
ax.set_ylabel("Second Principal component")
ax.set_xlim(-4,5)
ax.set_ylim(-3,2)

for i in range(len(pro1)):
    ax.text(pro1[i], pro2[i],data['Manufacturer'][i] )

ax2 = fig.add_subplot(111, label='2', frame_on=False)
names=['Horsepower', 'Length', 'Width', 'F.T.C.']
for i in range(len(vec)):
    ax2.arrow(0,0, vec[i,0], vec[i,1], color='r', head_width=0.02)
    if (i==0):
        ax2.text(vec[i,0]-0.155, vec[i,1]-0.07,names[i], color='r')
    else:
        ax2.text(vec[i,0]-0.155, vec[i,1],names[i], color='r')

ax2.axhline(0, ls='--', alpha=0.5)
ax2.axvline(0, ls='--', alpha=0.5)
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.set_xlim(-1,1)
ax2.set_ylim(-1,1)
plt.savefig('cars.png')
