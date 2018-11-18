#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// inverts a matrix A by turning first N columns of A|I into RREF
// # threads = 2N
// each thread corresponds to a particular column
// perform division on row to turn leading nonzero into a 1
// perform elimination on all other rows to make pivot column 0s
__global__ void MatrixInverse(double *A, int Ax, int Ay) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double mult;
	double to_mult;
	double old_val;
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
			mult = A[j*Ay + i]; // current row, pivot column
			to_mult = A[i*Ay + col]; //  pivot row, current column
			old_val = A[j*Ay + col]; // current row, current column
			if ((j != i)) {
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

__global__ void ExtractInverse(double *src, double* dst, int num_row, int num_col) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i % (2 * num_col) >= num_col) {
		dst[(num_row*(i / (2 * num_row))) + (i % (2 * num_row) - num_row)] = src[i];
	}


}

// adds arrays A and B and stores the result in C 
// assume all arrays have the same dimensions
__global__ void MatrixAdd(double * A, double * B, double * C) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	C[x] = A[x] + B[x];
}

// performs scalar multiplication on matrix A and scalar X
// stores result in B
__device__ void MatrixSMul(double * A, double * B, double scalar) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	B[x] = A[x] * scalar;
}

// transpose function, A is input, B is output, Ax and Ay are the dimensions of A
__global__ void MatrixTranspose(double * A, double * B, int Ax, int Ay) {
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

		// reset C array
		C[x] = 0;
		__syncthreads();

		int count;
		int Aindex, Bindex;
		double prod;
		for (count = 0; count < Ax; count++) {
			// row of C matrix
			Aindex = (x / Bx) * Ax + count;
			// column of C matrix
			Bindex = (x % Bx) + Bx * count;
			prod = A[Aindex] * B[Bindex];
			C[x] += prod;
		}
	}
}

__global__ void AddLambdaToDiagonal(double * A, double lambda, int Ax, int Ay) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	int my_row_pivot_index = (Ax * (x / Ax)) + (x / Ax);

	if (my_row_pivot_index == x) {
		A[x] = A[x] + lambda;
	}
}

// Function that appends a column of 1s to a matrix
// keeping new matrix in row major form
// constant time in parallel
// assume that dst has M x (N + 1)
__global__ void AppendOne(double* src, double* dst, int num_row, int num_col) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	int new_index = x + (x / num_col) + 1;

	dst[new_index] = src[x];

	if (new_index % (num_col + 1) == 1) {
		dst[new_index - 1] = 1;
	}
}

// takes an array of doubles and its dimensions as input
// sets the array to (((A^t)(A))^-1)(A^t)B
// where A is a matrix with Ay elements each having Ax features
// and B is a vector containing Ay elements
// C is a vector with Ax elements
void get_beta(double * A, double * B, double * C, int Ax, int Ay, double lambda) {
	int x;
	double * MatA = (double *)malloc(Ax * Ay * sizeof(double));
	double * MatA1 = (double *)malloc((Ax + 1) * Ay * sizeof(double));
	double * MatB = (double *)malloc((Ax + 1) * Ay * sizeof(double));
	double * MatC = (double *)malloc((Ax + 1) * (Ax + 1) * sizeof(double));
	double * MatD = (double *)malloc(2 * (Ax + 1) * (Ax + 1) * sizeof(double));
	double * MatA_d;
	double * MatA1_d;
	double * MatB_d;
	double * MatC_d;
	double * MatD_d;
	double * MatE_d;
	double * Beta_d;
	cudaMalloc((void **)&MatA_d, Ax * Ay * sizeof(double));
	cudaMalloc((void **)&MatA1_d, (Ax + 1) * Ay * sizeof(double));
	cudaMalloc((void **)&MatB_d, (Ax + 1) * Ay * sizeof(double));
	cudaMalloc((void **)&MatC_d, (Ax + 1) * (Ax + 1) * sizeof(double));
	cudaMalloc((void **)&MatD_d, 2 * (Ax + 1) * (Ax + 1) * sizeof(double));
	cudaMalloc((void **)&MatE_d, Ay * sizeof(double));
	cudaMalloc((void **)&Beta_d, (Ax + 1) * sizeof(double));
	cudaMemcpy(MatA_d, A, Ax * Ay * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(MatE_d, B, Ay * sizeof(double), cudaMemcpyHostToDevice);

	// Append 1s A
	AppendOne << < Ax, Ay >> > (MatA_d, MatA1_d, Ay, Ax);

	// Add new column
	Ax++;

	// B = Transpose(A)
	MatrixTranspose << < Ax, Ay >> > (MatA1_d, MatB_d, Ax, Ay);

	// C = BA
	MatrixMul << <Ax, Ax >> > (MatB_d, MatA1_d, MatC_d, Ay, Ax, Ax, Ay);

	// Regularization C = C - lambda*I
	AddLambdaToDiagonal << <Ax, Ax >> > (MatC_d, lambda, Ax, Ax);

	// Invert C
	MatrixAppendIdentity << <Ax, 2 * Ax >> > (MatC_d, MatD_d, Ax, Ax);
	MatrixInverse << <1, 2 * Ax >> > (MatD_d, Ax, 2 * Ax);
	ExtractInverse << <Ax, 2 * Ax >> > (MatD_d, MatC_d, Ax, Ax);

	// A = CB
	MatrixMul << <Ax, Ay >> > (MatC_d, MatB_d, MatA1_d, Ax, Ax, Ay, Ax);

	// Beta = AE
	// E is the known vector
	MatrixMul << <1, Ax >> > (MatA1_d, MatE_d, Beta_d, Ay, Ax, 1, Ay);

	// return Beta
	cudaMemcpy(C, Beta_d, Ax * sizeof(double), cudaMemcpyDeviceToHost);

	// free resources
	free(MatA);
	free(MatA1);
	free(MatB);
	free(MatC);
	free(MatD);
	cudaFree(MatA_d);
	cudaFree(MatA1_d);
	cudaFree(MatB_d);
	cudaFree(MatC_d);
	cudaFree(MatD_d);
	cudaFree(MatE_d);
	cudaFree(Beta_d);
}

// Performs matrix multiplication on A and B
// A a matrix of known values with Ay rows and Ax columns
// B is the beta vector with Ax values
// C is the output vector with Ay values
void linreg(double * A, double * B, double * C, int Ax, int Ay) {
	double * MatA = (double *)malloc(Ax * Ay * sizeof(double));
	double * MatA1 = (double *)malloc((Ax + 1) * Ay * sizeof(double));
	double * MatB = (double *)malloc((Ax + 1) * sizeof(double));
	double * MatC = (double *)malloc(Ay * sizeof(double));
	double * MatA_d;
	double * MatA1_d;
	double * MatB_d;
	double * MatC_d;
	cudaMalloc((void **)&MatA_d, Ax * Ay * sizeof(double));
	cudaMalloc((void **)&MatA1_d, (Ax + 1) * Ay * sizeof(double));
	cudaMalloc((void **)&MatB_d, (Ax + 1) * sizeof(double));
	cudaMalloc((void **)&MatC_d, Ay * sizeof(double));
	cudaMemcpy(MatA_d, A, Ax * Ay * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(MatB_d, B, (Ax + 1) * sizeof(double), cudaMemcpyHostToDevice);

	// Append 1s to A
	AppendOne << <Ax, Ay >> > (MatA_d, MatA1_d, Ay, Ax);

	// Add a column
	Ax++;

	// C = AB
	MatrixMul << <1, Ay >> > (MatA1_d, MatB_d, MatC_d, Ax, Ay, 1, Ax);

	// return C
	cudaMemcpy(C, MatC_d, Ay * sizeof(double), cudaMemcpyDeviceToHost);

	// free resources
	free(MatA);
	free(MatA1);
	free(MatB);
	free(MatC);
	cudaFree(MatA_d);
	cudaFree(MatA1_d);
	cudaFree(MatB_d);
	cudaFree(MatC_d);
}