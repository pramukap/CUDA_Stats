import numpy as np
import ctypes
from ctypes import *

LIBPATH = ''

class LogisticRegression:
    def init(self, lr=0.01, n_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.lib = ctypes.CDLL(LIBPATH, mode=ctypes.RTLD_GLOBAL)
    
    def __add_intercept(self, X):
        pass

    def __sigmoid(self, z):
        func = self.lib.vec_sigmoid
        func.argtypes = [POINTER(c_double), POINTER(c_double), c_size_t, c_size_t]
        ret_val = np.zeros(z.shape)
        z_p = z.ctypes.data_as(POINTER(c_double))
        ret_val_p = ret_val.ctypes.data_as(POINTER(c_double))
        func(z_p, ret_val_p, 1, len(z))
        return ret_val
    
    def __loss(self, h, y):
        pass

    def fit(self, X, y):
        pass
    
    def predict_prob(self, X):
        pass
    
    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold
    
