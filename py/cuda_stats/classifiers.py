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
