
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// define matrix size
#define X 20
#define Y 1000

// possible matrix struct, didnt use here
struct Matrix {

	int col;
	int row;
	double * data;
};

// transpose function, A is input, B is output, Ax and Ay are the dimensions of A
__device__ void MatrixTranspose(double * A, double * B, int Ax, int Ay) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = threadIdx.x;
	int new_row, new_loc;
	if (x == 0) {
		new_loc = 0;
	}
	else {
		new_row = (x % Ax) * Ay;
		new_loc = new_row + (x / Ax);
	}
	
	B[new_loc] = A[x];
}

// kernel that calls the function
__global__ void MatrixKernel(double * A, double * B, int Ax, int Ay) {
	MatrixTranspose(A, B, Ax, Ay);
}

int main()
{
	int size = X * Y * sizeof(double);
	int arrSize = X * Y;
	double * MatA = (double *)malloc(size);
	double * MatB = (double *)malloc(size);
	double * MatA_d;
	double * MatB_d;
	
	cudaMalloc((void **)&MatA_d, size);
	cudaMalloc((void **)&MatB_d, size);

	int x;
	for (x = 0; x < arrSize; x++) {
		MatA[x] = x;
		printf("%d ", (int)MatA[x]);
		if (x != 0) {
			if ((x % X) == (X - 1)) {
				printf("\n");
			}
		}
	}
	printf("\n");
	cudaMemcpy(MatA_d, MatA, size, cudaMemcpyHostToDevice);

	MatrixKernel << <X, Y >> > (MatA_d, MatB_d, X, Y);

	cudaMemcpy(MatB, MatB_d, size, cudaMemcpyDeviceToHost);

	for (x = 0; x < arrSize; x++) {
		printf("%d ", (int)MatB[x]);
		if (x != 0) {
			if ((x % Y) == (Y - 1)) {
				printf("\n");
			}
		}
	}

    return 0;
}

