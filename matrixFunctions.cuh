#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void MatrixAdd(double * A, double * B, double * C);
__global__ void MatrixSMul(double * A, double * B, double X);
__global__ void MatrixTranspose(double * A, double * B, int Ax, int Ay);
__global__ void MatrixMul(double * A, double * B, double * C, int Ax, int Ay, int Bx, int By);
__global__ void MatrixInverse(double *A, int Ax, int Ay);
__global__ void MatrixAppendIdentity(double* src, double* dst, int num_row, int num_col);
__global__ void ExtractInverse(double *src, double* dst, int num_row, int num_col);
__global__ void AppendOne(double* src, double* dst, int num_row, int num_col);
__global__ void AddLambdaToDiagonal(double * A, double lambda, int Ax, int Ay);

void get_beta(double * A, double * B, double * C, int Ax, int Ay, double lambda);
void linreg(double * A, double * B, double * C, int Ax, int Ay);
