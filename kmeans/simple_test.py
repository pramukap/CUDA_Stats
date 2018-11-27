import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from clustering import KMeans

data = np.array([1,1,0,0,0,1,1,0,5,5,5,7,7,5,6,6,0,9,0,8,1,8,1,9])
data = np.reshape(data, (12,2))

model = KMeans(k=3, iterations=10)
model.fit(data)
print(model.centroids_)
