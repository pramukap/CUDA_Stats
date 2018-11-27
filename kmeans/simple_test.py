import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from clustering import KMeans

data = np.array([1,1,0,0,0,1,1,0,5,5,5,7,7,5,6,6,0,9,0,8,1,8,1,9])
data = np.reshape(data, (12,2))

print(data)

model = KMeans(k=3, iterations=10)
model.fit(data)
print(model.centroids_)
labels = model.label(data)

print(labels)

data = pd.DataFrame(data)
print(data)

for ki in range(3):
    plt.scatter(data[labels==ki][0], data[labels==ki][1])

for c in model.centroids_:
    plt.scatter(c[0],c[1],c='k',marker='x')

plt.show()
