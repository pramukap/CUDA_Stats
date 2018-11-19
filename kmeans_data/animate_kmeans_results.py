import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


labels = pd.read_csv('iris_labels.csv', header=None)
labels = np.transpose(labels)
labels

kmean0 = pd.read_csv('kmean0.csv', header=None)
kmean1 = pd.read_csv('kmean1.csv', header=None)
kmean2 = pd.read_csv('kmean2.csv', header=None)


iris = pd.read_csv('iris.csv', header=None)

fig, ax = plt.subplots()
xdata, ydata = [], []
scat = plt.scatter([], [], animated=True)

def init():
    return scat,

def update(frame):
    plt.clf();
    print(frame)
    scat = plt.scatter(iris.loc[labels[frame]==0,0],iris.loc[labels[frame]==0,1],c='orange')
    scat = plt.scatter(iris.loc[labels[frame]==1,0],iris.loc[labels[frame]==1,1],c='blue')
    scat = plt.scatter(iris.loc[labels[frame]==2,0],iris.loc[labels[frame]==2,1],c='r')
    scat = plt.scatter(kmean0.iloc[frame,0], kmean0.iloc[frame,1], c='black',marker='X',s=160)
    scat = plt.scatter(kmean1.iloc[frame,0], kmean1.iloc[frame,1], c='black',marker='X', s=160)
    scat = plt.scatter(kmean2.iloc[frame,0], kmean2.iloc[frame,1], c='black', marker='X',s=160)
    return scat,

ani = FuncAnimation(fig, update, frames=range(12),
                    init_func=init, blit=False)
#plt.show()

ani.save('kmeans.mp4')
