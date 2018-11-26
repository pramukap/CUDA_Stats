import numpy as np
import ctypes
from ctypes import *

LIBPATH = './log_reg.so'

class PyLogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                #print(f'loss: {self.__loss(h, y)} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold

class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.lib = ctypes.CDLL(LIBPATH, mode=ctypes.RTLD_GLOBAL)
    
    def fit(self, X, y):
        assert X.shape[0] == y.shape[0], "Dimensions do not match"
        if not hasattr(self, 'theta'):
            self.theta = np.zeros(X.shape[1]).astype('float64')
        func = self.lib.fit
        func.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_double, c_size_t, c_size_t, c_size_t]

        Xp = X.ctypes.data_as(POINTER(c_double))
        yp = y.ctypes.data_as(POINTER(c_double))
        thetap = self.theta.ctypes.data_as(POINTER(c_double))

        func(Xp, yp, thetap, self.lr, X.shape[0], X.shape[1], self.n_iter)

    def predict_prob(self, X):
        assert X.shape[1] == self.theta.shape[0], "Dimensions do not match"
        func = self.lib.predict_proba
        func.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_size_t, c_size_t]

        y = np.zeros(X.shape[0]).astype('float64')

        Xp = X.ctypes.data_as(POINTER(c_double))
        yp = y.ctypes.data_as(POINTER(c_double))
        thetap = self.theta.ctypes.data_as(POINTER(c_double))
        func(Xp, thetap, yp, X.shape[0], X.shape[1])
        return y
    
    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold

if __name__ == "__main__":
    lib = ctypes.CDLL(LIBPATH, mode=ctypes.RTLD_GLOBAL)
    fit = lib.fit
    # void    fit(double *X, double *y, double *theta, double lr, size_t n, size_t m, size_t n_iter)
    fit.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_double, c_size_t, c_size_t, c_size_t]
    mat = np.asarray([[i for i in range(1024)] for _ in range(1024)]).astype('float64')
    y = np.asarray([i % 2 == 0 for i in range(1024)]).astype('float64')
    theta = np.asarray([0 for _ in range(1024)]).astype('float64')
    m_p = mat.ctypes.data_as(POINTER(c_double))
    y_p = mat.ctypes.data_as(POINTER(c_double))
    theta_p = mat.ctypes.data_as(POINTER(c_double))
    fit(m_p, y_p, theta_p, 0.01, 1024, 1024, 1000)
    print(theta[:24])
