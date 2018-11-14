
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// define matrix size
#define AX 3
#define AY 5
#define BX 5
#define BY 3


// possible matrix struct, didnt use here
struct Matrix {

	int col;
	int row;
	double * data;
};

// inverts a matrix A by turning first N columns of A|I into RREF
// # threads = 2N

//TODO: write rule for swapping
//TODO: write function to concatenate identity matrix to the end of A
//              effectively     A:-> A|I

// each thread corresponds to a particular column

// perform division on row to turn leading nonzero into a 1
// perform elimination on all other rows to make pivot column 0s
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
			//printf("mult = %f index = %d, to_mult = %f index = %d, old_val = %f index = %d, thread = %d, j = %d, i = %d, col = %d, Ay = %d\n", mult, (j*Ay + i), to_mult, (i*Ay + col), old_val, (j*Ay + col), col, j, i, col, Ay);
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
			Bindex = (x % Bx) +  Bx * count;
			prod = A[Aindex] * B[Bindex];
			C[x] += prod;
		}
	}
}

// takes an array of doubles and its dimensions as input
// sets the array to (((A^t)(A))^-1)(A^t)
void get_normal(double * A, int Ax, int Ay) {
	int x;
	double * MatB = (double *)malloc(Ax * Ay * sizeof(double));
	double * MatC = (double *)malloc(Ax * Ax * sizeof(double));
	double * MatD = (double *)malloc(2 * Ax * Ax * sizeof(double));
	double * MatA_d;
	double * MatB_d;
	double * MatC_d;
	double * MatD_d;
	cudaMalloc((void **)&MatA_d, Ax * Ay * sizeof(double));
	cudaMalloc((void **)&MatB_d, Ax * Ay * sizeof(double));
	cudaMalloc((void **)&MatC_d, Ax * Ax * sizeof(double));
	cudaMalloc((void **)&MatD_d, 2 * Ax * Ax * sizeof(double));
	cudaMemcpy(MatA_d, A, Ax * Ay * sizeof(double), cudaMemcpyHostToDevice);

	// B = Transpose(A)
	MatrixTranspose << < Ay, Ax >> > (MatA_d, MatB_d, Ax, Ay);
	cudaMemcpy(MatB, MatB_d, Ax * Ay * sizeof(double), cudaMemcpyDeviceToHost);
	printf("[At] = \n");
	for (x = 0; x < (Ax * Ay); x++) {
		printf("%f ", MatB[x]);
		if (x != 0) {
			if ((x % Ay) == (Ay - 1)) {
				printf("\n");
			}
		}
	}
	printf("\n");

	// C = BA
	MatrixMul << <Ax, Ax >> > (MatB_d, MatA_d, MatC_d, Ay, Ax, Ax, Ay);
	cudaMemcpy(MatC, MatC_d, Ax * Ax * sizeof(double), cudaMemcpyDeviceToHost);
	printf(" [At][A] = \n");
	for (x = 0; x < (Ax * Ax); x++) {
		printf("%f ", MatC[x]);
		if (x != 0) {
			if ((x % Ax) == (Ax - 1)) {
				printf("\n");
			}
		}
	}
	printf("\n");

	// Invert C
	MatrixAppendIdentity << <Ax, 2 * Ax >> > (MatC_d, MatD_d, Ax, Ax);
	MatrixInverse << <1, 2 * Ax >> > (MatD_d, Ax, 2 * Ax);
	ExtractInverse << <Ax, 2 * Ax >> > (MatD_d, MatC_d, Ax, Ax);
	cudaMemcpy(MatC, MatC_d, Ax * Ax * sizeof(double), cudaMemcpyDeviceToHost);
	printf(" ([At][A])^-1 = \n");
	for (x = 0; x < (Ax * Ax); x++) {
		printf("%f ", MatC[x]);
		if (x != 0) {
			if ((x % (Ax)) == ((Ax) - 1)) {
				printf("\n");
			}
		}
	}
	printf("\n");

	// A = CB
	MatrixMul << <Ay, Ax >> > (MatC_d, MatB_d, MatA_d, Ax, Ax, Ay, Ax);
	cudaMemcpy(A, MatA_d, Ax * Ay * sizeof(double), cudaMemcpyDeviceToHost);
}

int main()
{
	int Asize = AX * AY * sizeof(double);
	int Bsize = BX * BY * sizeof(double);
	int Csize = AX * BY * sizeof(double);
	int AarrSize = AX * AY;
	int BarrSize = BX * BY;
	int CarrSize = BX * AY;
	double * MatA = (double *)malloc(Asize);
	double * MatB = (double *)malloc(Bsize);
	double * MatC = (double *)malloc(Csize);
	double * MatD = (double *)malloc(2 * Asize);
	double * MatA_d;
	double * MatB_d;
	double * MatC_d;
	double * MatD_d;
	
	cudaMalloc((void **)&MatA_d, Asize);
	cudaMalloc((void **)&MatB_d, Bsize);
	cudaMalloc((void **)&MatC_d, Csize);
	cudaMalloc((void **)&MatD_d, 2 * Asize);

	// set up array
	double Mat[15] = { 1, 2, 3, 0, 1, 4, 5, 6, 0, 8, 4, 7, 1, 2, 5};
	double Matb[15] = { 1, 0, 5, 8, 1, 2, 1, 6, 4, 2, 3, 4, 0, 7, 5 };
	memcpy(MatA, Mat, 15 * sizeof(double));
	memcpy(MatB, Matb, 15 * sizeof(double));
	cudaMemcpy(MatA_d, MatA, AX * AY * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(MatB_d, MatB, BY * BY * sizeof(double), cudaMemcpyHostToDevice);

	// print initial array
	int x;
	printf("[A] = \n");
	for (x = 0; x < AarrSize; x++) {
		printf("%d ", (int)MatA[x]);
		if (x != 0) {
			if ((x % AX) == (AX - 1)) {
				printf("\n");
			}
		}
	}
	printf("\n");

	get_normal(MatA, AX, AY);

	printf("Final Array = \n");
	for (x = 0; x < AarrSize; x++) {
		printf("%f ", MatA[x]);
		if (x != 0) {
			if ((x % AY) == (AY - 1)) {
				printf("\n");
			}
		}
	}
	printf("\n");

    return 0;
}

