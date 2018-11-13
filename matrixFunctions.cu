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
__global__ void MatrixMul(double * A, double * B, double * C, int Ax, int Ay, int Bx, int By) {
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

// inverts a matrix A by turning first N columns of A|I into RREF
// # threads = 2N

//TODO: write rule for swapping
//TODO: write function to concatenate identity matrix to the end of A
//              effectively     A:-> A|I

// each thread corresponds to a particular column

// perform division on row to turn leading nonzero into a 1
// perform elimination on all other rows to make pivot column 0s
// call so Ax = AY and Ay = 2 * AX
__global__ void MatrixInverse(double *A, int Ax, int Ay) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double mult;
	double to_mult;
	double old_val;
	int current_pivot_col = 0;
	int i = 0;
	for (i = 0; i < Ax; i++) {
		// SWAP CODE
		if (i == col && A[i*Ay + col] == 0) {
			for (int k = i; k < Ax; k++) {
				if (A[k*Ay + col] != 0) {
					for (int x = 0; x < Ay; x++) {
						int tmp = A[i*Ay + x];
						A[i*Ay + x] = A[k*Ay + x];
						A[k*Ay + x] = tmp;
					}
					break;
				}

			}
		}

		// divide element by pivot
		__syncthreads();
		A[i*Ay + col] = A[i*Ay + col] / A[i*Ay + i];
		__syncthreads();

		for (int j = 0; j < Ax; j++) {
			mult = A[j*Ay + i];
			to_mult = A[i*Ay + col];
			old_val = A[j*Ay + col];
			if ((j != i) && (A[j*Ay + i] != 0)) {
				A[j*Ay + col] = old_val - mult * to_mult;
			}
		}

		__syncthreads();
	}
}

// Function that appends an identity matrix to the right of the current matrix
// keeping new matrix in row major form
// constant time in parallel
// assume that dst has 2*N*N = 2*len(src) allocated
__global__ void MatrixAppendIdentity(double* src, double* dst, int num_row, int num_col) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i % (2 * num_col) < num_col) {
		dst[i] = src[(num_row*(i / (2 * num_row))) + (i % (2 * num_row))];
	}
	else if ((i % (2 * num_row) - num_row == i / (2 * num_row))) {
		dst[i] = 1;
	}
	else {
		dst[i] = 0;
	}

}



__global__ void ExtractInverse(double *src, double* dst, int num_row, int num_col){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i % (2*num_col) >= num_col){
		dst[(num_row*(i / (2 * num_row))) + (i % (2 * num_row) - num_row)] = src[i];
    }


}







