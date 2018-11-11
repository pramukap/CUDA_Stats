#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__device__ void MatrixAdd(double * A, double * B, double * C);
__device__ void MatrixSMul(double * A, double * B, double X);
__device__ void MatrixTranspose(double * A, double * B, int Ax, int Ay);
__device__ void MatrixMul(double * A, double * B, double * C, int Ax, int Ay, int Bx, int By);
__global__ void MatrixInverse(double *A, int Ax, int Ay);
__global__ void MatrixAppendIdentity(double* src, double* dst, int num_row, int num_col);
__global__ void ExtractInverse(double *src, double* dst, int num_row, int num_col);
