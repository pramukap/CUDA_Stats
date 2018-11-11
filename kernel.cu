
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "matrixFunctions.cuh"
#include <stdio.h>

// define matrix size
#define X	3
#define Y	3 

// possible matrix struct, didnt use here
struct Matrix {

	int col;
	int row;
	double * data;
};

// kernel that calls the function
__global__ void MatrixKernel(double * A, double * B, int Ax, int Ay) {
//	MatrixAppendIdentity(A, B, Ax, Ay);
//	MatrixInverse(B, 2*Ax, Ay);
//	ExtractInverse(B, A, Ax, Ay);
}

int main()
{
	int size = X * Y * sizeof(double);
	int arrSize = X * Y;
	double * MatA = (double *)malloc(size);
	double * MatB = (double *)malloc(2 * size);
	double * MatA_d;
	double * MatB_d;
	
	cudaMalloc((void **)&MatA_d, size);
	cudaMalloc((void **)&MatB_d, 2 * size);

	double Mat[9] = {1, 2, 3, 0, 1, 4, 5, 6, 0};
	memcpy(MatA, Mat, 9);
	
	int x;
	for (x = 0; x < arrSize; x++) {
		//MatA[x] = x;
		printf("%d ", (int)MatA[x]);
		if (x != 0) {
			if ((x % X) == (X - 1)) {
				printf("\n");
			}
		}
	}
	printf("\n");
	cudaMemcpy(MatA_d, MatA, size, cudaMemcpyHostToDevice);

	//MatrixKernel << <X, Y >> > (MatA_d, MatB_d, X, Y);
	MatrixAppendIdentity <<<X, 2*Y>>> (MatA_d, MatB_d, X, Y);
	MatrixInverse <<<1, 2*X>>> (MatB_d, 2*X, Y);
	ExtractInverse <<<X, 2*Y>>> (MatB_d, MatA_d, X, Y);

	cudaMemcpy(MatA, MatA_d, size, cudaMemcpyDeviceToHost);

	for (x = 0; x < arrSize; x++) {
		printf("%d ", (int)MatA[x]);
		if (x != 0) {
			if ((x % Y) == (Y - 1)) {
				printf("\n");
			}
		}
	}

    return 0;
}

