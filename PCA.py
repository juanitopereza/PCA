import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%

data = pd.read_csv("Cars93.csv")

hp = (data['Horsepower'] - data['Horsepower'].mean())/data['Horsepower'].std()
price = (data['Price'] - data['Price'].mean())/data['Price'].std()

M = np.vstack((hp,price))

cova = np.cov(M)

val, vec = np.linalg.eig(cova)

proy_pc1 = M.T.dot(vec[:,0])
proy_pc2 = M.T.dot(vec[:,1])


x = np.linspace(hp.min(),hp.max())

#%%
plt.figure(figsize=(10,10))
# plt.scatter(hp,price)
# plt.plot(x, (vec[1,0]/vec[0,0])*x, 'r')
# plt.plot(x, (vec[1,1]/vec[0,1])*x, '--b')
# plt.xlim(-4,4)
# plt.ylim(-4,4)

#%%
plt.figure(figsize=(10,10))
plt.scatter(data['Horsepower'], data['Price'])
plt.plot(x*data['Horsepower'].std() + data['Horsepower'].mean(), (vec[1,0]/vec[0,0])*x*data['Price'].std() + data['Price'].mean(), 'r')
plt.plot(x*data['Horsepower'].std() + data['Horsepower'].mean(), (vec[1,1]/vec[0,1])*x*data['Price'].std() + data['Price'].mean(), '--b')

#%%

mat = np.vstack((proy_pc1,np.zeros(len(proy_pc1))))
points = vec.dot(mat)
np.shape(points)

plt.figure(figsize=(10,10))
plt.scatter(hp,price)
plt.plot(x, (vec[1,0]/vec[0,0])*x, 'r')
plt.scatter(points[0,:], points[1,:], marker='x',c='k')
for i in range(len(hp)):
    plt.plot([points[0,i], hp[i]], [points[1,i], price[i]], '--k')
plt.show()

#%%
xx = np.linspace(-3,5)

plt.figure(figsize=(10,10))
plt.scatter(proy_pc1,proy_pc2, s=20)
plt.scatter(proy_pc1,np.zeros(len(proy_pc1)), marker='x', c='k')
plt.plot(xx,np.zeros(len(xx)), 'r')
for i in range(len(proy_pc1)):
    plt.vlines(proy_pc1[i],0,proy_pc2[i], linestyles='--')
plt.xlim(-2.5,4.5)
plt.show()
