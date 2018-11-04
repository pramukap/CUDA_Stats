#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// adds arrays A and B and stores the result in C 
// assume all arrays have the same dimensions
__device__ void MatrixAdd(double * A, double * B, double * C) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	C[x] = A[x] + B[x];
}

// performs scalar multiplication on matrix A and scalar X
// stores result in B
__device__ void MatrixSMul(double * A, double * B, double X) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	B[x] = A[x] * X;
}

// transpose function, A is input, B is output, Ax and Ay are the dimensions of A
__device__ void MatrixTranspose(double * A, double * B, int Ax, int Ay) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
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

// multiplies the matrices A and B and stores them into C
// Ax, Ay, Bx, By are the dimensions
// use a thread for each element of the final C array.
__device__ void MatrixMul(double * A, double * B, double * C, int Ax, int Ay, int Bx, int By) {
	if (Ax == By) {
		// total array position
		int x = blockIdx.x * blockDim.x + threadIdx.x;

		int count;
		int Aindex, Bindex;
		double prod;
		for (count = 0; count < Ax; count++) {
			// row of C matrix
			Aindex = (x / Bx) * Ax + count;
			// column of C matrix
			Bindex = (x % Bx) + By * count;
			prod = A[Aindex] * B[Bindex];
			C[x] += prod;
		}
	}
}
