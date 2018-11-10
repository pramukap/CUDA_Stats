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


// inverts a matrix A by turning first N columns of A|I into RREF
// # threads = 2N

//TODO: write rule for swapping
//TODO: write function to concatenate identity matrix to the end of A
//              effectively     A:-> A|I

// each thread corresponds to a particular column

// perform division on row to turn leading nonzero into a 1
// perform elimination on all other rows to make pivot column 0s
__device__ void MatrixInverse(double *A, int Ax, int Ay){

    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int current_pivot_col = 0;

    for(int i = 0; i < Ax; i++){
        a[i*Ax + col] = a[i*Ax + col] / a[i*Ax + i];

        __syncthreads();
        
        for(int j = 0; j < Ax; i++){
            if(j != i){
                a[j*Ax+col] = A[j*Ax+col] - A[j*Ax + i]*A[i*Ax + col];
            }
        }

        __syncthreads();

    }

}

// Function that appends an identity matrix to the right of the current matrix
// keeping new matrix in row major form
// constant time in parallel
// assume that dst has 2*N*N = 2*len(src) allocated
__device__ void MatrixAppendIdentity(double* src, double* dst, int num_row, int num_col){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i % (2 * num_col) < num_col){
        dst[i] = x[(num_row*(i/(2*num_row)))+ (i % (2*num_row))];
    } else if ((i%(2*num_row) - num_row == i / (2*num_row)) {
        dst[i] = 1;
    } else {
        dst[i] = 0;
    }

}











