import ctypes
from ctypes import *
import numpy

dylib = ctypes.CDLL('libprint.so')
ndarr = numpy.asarray([[float(i) for i in range(24)] for _ in range(24)]).astype('float64')
func = dylib.print_matrix
func.argtypes = [POINTER(c_double), c_size_t, c_size_t]
func(ndarr.ctypes.data_as(POINTER(c_double)), 24, 24)