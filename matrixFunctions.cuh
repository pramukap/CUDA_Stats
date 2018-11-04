#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__device__ void MatrixAdd(double * A, double * B, double * C);
__device__ void MatrixSMul(double * A, double * B, double X);
__device__ void MatrixTranspose(double * A, double * B, int Ax, int Ay);
__device__ void MatrixMul(double * A, double * B, double * C, int Ax, int Ay, int Bx, int By);
