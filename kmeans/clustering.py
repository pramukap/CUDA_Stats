import ctypes
from ctypes import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


class KMeans:

    def __init__(self, k=3, iterations=10, timing=False):
        self.k_ = k
        self.itr_ = iterations
        self.centroids_ = None
        self.dylib = ctypes.CDLL('./obj/kmeans.so')
        print('initalized kmeans with k = ',k,', itr = ',iterations)
        self.timeflag = timing

    def fit(self, data):
        data = np.array(data, dtype=float)
        m = data.shape[0]
        if(len(data.shape)>1):
            n = data.shape[1]
        else:
            n = 1
        print('mxn: ',m,'x',n)
        data = np.reshape(data,(m*n,))
        data = data.astype(c_double)
        self.centroids_ = np.zeros(self.k_*n).astype(c_double)
        func = self.dylib.kmeans
        func.argtypes = [POINTER(c_double), c_size_t, c_size_t, c_size_t, POINTER(c_double), c_size_t]
        if(self.timeflag):
            start = time.time()
        func(data.ctypes.data_as(POINTER(c_double)), m, n, self.k_, self.centroids_.ctypes.data_as(POINTER(c_double)), self.itr_)
        if(self.timeflag):
            end = time.time()
            print('Time to fit model: ', end - start)

        self.centroids_ = np.reshape(self.centroids_, (self.k_, n))
        return self.centroids_


    def label(self, data):
        assert self.centroids_.any(), 'Must call fit(X) before label()'
        data = np.array(data, dtype=float)
        m = data.shape[0]
        if(len(data.shape)>1):
            n = data.shape[1]
        else:
            n = 1
        data = np.reshape(data,(m*n,))
        data = data.astype(c_double)

        labels = np.zeros(m).astype(c_int)

        func = self.dylib.kmeans_classify
        func.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_int), c_size_t, c_size_t, c_size_t]
        if(self.timeflag):
            start = time.time()
        func(self.centroids_.ctypes.data_as(POINTER(c_double)), data.ctypes.data_as(POINTER(c_double)), labels.ctypes.data_as(POINTER(c_int)), m, n, self.k_)
        if(self.timeflag):
            end = time.time()
            print('Time to label datapoints: ', end - start)
        return labels
'''
data = np.array(pd.read_csv('iris.csv', header=None))
model = KMeansClustering()
model.fit(data)
print(model.label(data))
'''
