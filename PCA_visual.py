import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("USArrests.csv")

Murder = (data['Murder'] - data['Murder'].mean())/data['Murder'].std()
Assault = (data['Assault'] - data['Assault'].mean())/data['Assault'].std()
UrbanPop = (data['UrbanPop'] - data['UrbanPop'].mean())/data['UrbanPop'].std()
Rape = (data['Rape'] - data['Rape'].mean())/data['Rape'].std()


data.columns[0]

M = np.vstack((Murder,Assault,UrbanPop,Rape))

M[3].std()

cov = np.cov(M)

cov

val, vec = np.linalg.eig(cov)

val
vec


pro1 = vec[:,0].dot(M)
pro2 = vec[:,1].dot(M)

#%%

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, label='1')

ax.scatter(pro1, -pro2, alpha=0)
ax.set_xlabel("First Principal component")
ax.set_ylabel("Second Principal component")
ax.set_xlim(-4,4)
ax.set_ylim(-3,3)

for i in range(len(pro1)):
    ax.text(pro1[i], -pro2[i],data.iloc[i][0] )

ax2 = fig.add_subplot(111, label='2', frame_on=False)
for i in range(len(vec)):
    ax2.arrow(0,0, vec[i,0], -vec[i,1], color='r', head_width=0.02)
    ax2.text(vec[i,0]+0.03, -vec[i,1],data.columns[i+1], color='r')

ax2.axhline(0, ls='--', alpha=0.5)
ax2.axvline(0, ls='--', alpha=0.5)
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.set_xlim(-1,1)
ax2.set_ylim(-1,1)
plt.show()
