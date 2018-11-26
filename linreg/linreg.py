import ctypes
from ctypes import *
import numpy as np

LIBPATH='./lin_reg.so'

def LinearRegression(object):
    def __init__(self, l=0.0655):
        self.lib = ctypes.cdll.LoadLibrary(LIBPATH)
        self.l = l

    # get_beta(double * A, double * B, double * C, int Ax, int Ay, double lambda)
    def fit(self, X, y):
        assert X.shape[0] == y.shape[0], 'Dimensions do not match'

        if not hasattr(self, 'coeffs'):
            self.coeffs = np.zeros(X.shape[1]).astype('float64')

        func = self.lib.get_beta
        func.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int, c_double]

        Xp = X.ctypes.data_as(POINTER(c_double))
        yp = yp.ctypes.data_as(POINTER(c_double))
        cp = coeffs.ctypes.data_as(POINTER(c_double))

        func(Xp, yp, cp, X.shape[1], X.shape[0], self.l)
    
    # linreg(double * A, double * B, double * C, int Ax, int Ay)
    def predict(self, X):
        assert X.shape[1] == self.coeffs.shape[0], 'Dimensions do not match'

        func = self.lib.linreg
        func.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), X.shape[1], X.shape[0]]

        y = np.zeros(X.shape[0]).astype('float64')

        Xp = X.ctypes.data_as(POINTER(c_double))
        yp = y.ctypes.data_as(POINTER(c_double))
        cp = self.coeffs.ctypes.data_as(POINTER(c_double))

        func(Xp, yp, cp, X.shape[1], X.shape[0])

        return y
