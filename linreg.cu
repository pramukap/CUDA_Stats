
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "house_med.h"

// define matrix size
#define AX 8
#define AY 1000
#define BX 1
#define BY 1000

// inverts a matrix A by turning first N columns of A|I into RREF
// # threads = 2N
// each thread corresponds to a particular column
// perform division on row to turn leading nonzero into a 1
// perform elimination on all other rows to make pivot column 0s
// O(Ax^2) time
// O(Ay) work
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
			mult = A[j*Ay + i]; // current row, pivot column
			to_mult = A[i*Ay + col]; //  pivot row, current column
			old_val = A[j*Ay + col]; // current row, current column
			if ((j != i)) {
				A[j*Ay + col] = old_val - mult * to_mult;
			}
			//if (col == 20) {
				//printf("mult = %fl index = %d, to_mult = %fl index = %d, old_val = %fl index = %d, j = %d, i = %d, col = %d, new = %fl\n", mult, (j*Ay + i), to_mult, (i*Ay + col), old_val, (j*Ay + col), j, i, col, A[j*Ay + col]);
			//}
		}

		__syncthreads();
	}

	if (col == 0) {
		printf("Finished inversion = \n");
		for (i = 0; i < Ax * Ay; i++) {
			printf("%fl ", A[i]);
			if (i != 0) {
				if ((i % Ay) == (Ay - 1)) {
					printf("\n");
				}
			}
		}
		printf("\n");
	}
}

// Function that appends an identity matrix to the right of the current matrix
// keeping new matrix in row major form
// constant time in parallel
// assume that dst has 2*N*N = 2*len(src) allocated
// O(1) time
// O(Ax * Ay) work
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

// Extracts the inverse matrix from the identity matrix
// O(1) time
// O(Ax * Ay) work
__global__ void ExtractInverse(double *src, double* dst, int num_row, int num_col) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i % (2 * num_col) >= num_col) {
		dst[(num_row*(i / (2 * num_row))) + (i % (2 * num_row) - num_row)] = src[i];
	}
}

// adds arrays A and B and stores the result in C 
// assume all arrays have the same dimensions
// O(1) time
// O(Ax * Ay) work
__global__ void MatrixAdd(double * A, double * B, double * C) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	C[x] = A[x] + B[x];
}

// performs scalar multiplication on matrix A and scalar X
// stores result in B
// O(1) time
// O(Ax * Ay) work
__device__ void MatrixSMul(double * A, double * B, double scalar) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	B[x] = A[x] * scalar;
}

// Transpose function, A is input, B is output, Ax and Ay are the dimensions of A
// O(1) time
// O(Ax * Ay)
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

// Multiplies the matrices A and B and stores them into C
// Ax, Ay, Bx, By are the dimensions
// Use a thread for each element of the final C array
// O(Ax) time
// O(Bx * Ay) work
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
			//printf("Aindex = %d, Bindex = %d, prod = %f, x = %d, C[x] = %f\n", Aindex, Bindex, prod, x, C[x]);
		}
	}
}

// Adds the value lambda to the diagonal of the input matrix
// O(1) time
// O(Ax * Ay) work
__global__ void AddLambdaToDiagonal(double * A, double lambda, int Ax, int Ay) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;

		int my_row_pivot_index = (Ax * (x/Ax)) + (x/Ax);

		if  (my_row_pivot_index == x) {
			A[x] = A[x] + lambda;
		}
}

// Function that appends a column of 1s to a matrix
// keeping new matrix in row major form
// constant time in parallel
// assume that dst has M x (N + 1)
// O(1) time
// O(Ax * Ay) work
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
// O(Ay) time
// O(Ax * Ay) work
void get_beta(double * A, double * B, double * C, int Ax, int Ay, double lambda) {
	int x;
	double * MatA = (double *)malloc(Ax * Ay * sizeof(double));
	double * MatA1 = (double *)malloc((Ax + 1) * Ay * sizeof(double));
	double * MatB = (double *)malloc((Ax + 1) * Ay * sizeof(double));
	double * MatC = (double *)malloc((Ax + 1) * (Ax + 1) * sizeof(double));
	double * MatC2 = (double *)malloc((Ax + 1) * (Ax + 1) * sizeof(double));
	double * MatD = (double *)malloc(2 * (Ax + 1) * (Ax + 1) * sizeof(double));
	double * MatA_d;
	double * MatA1_d;
	double * MatB_d;
	double * MatC_d;
	double * MatC2_d;
	double * MatC3_d;
	double * MatD_d;
	double * MatE_d;
	double * Beta_d;
	cudaMalloc((void **)&MatA_d, Ax * Ay * sizeof(double));
	cudaMalloc((void **)&MatA1_d, (Ax + 1) * Ay * sizeof(double));
	cudaMalloc((void **)&MatB_d, (Ax + 1) * Ay * sizeof(double));
	cudaMalloc((void **)&MatC_d, (Ax + 1) * (Ax + 1) * sizeof(double));
	cudaMalloc((void **)&MatC2_d, (Ax + 1) * (Ax + 1) * sizeof(double));
	cudaMalloc((void **)&MatC3_d, (Ax + 1) * (Ax + 1) * sizeof(double));
	cudaMalloc((void **)&MatD_d, 2 * (Ax + 1) * (Ax + 1) * sizeof(double));
	cudaMalloc((void **)&MatE_d, Ay * sizeof(double));
	cudaMalloc((void **)&Beta_d, (Ax + 1) * sizeof(double));
	cudaMemcpy(MatA_d, A, Ax * Ay * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(MatE_d, B, Ay * sizeof(double), cudaMemcpyHostToDevice);

	// Append 1s A
	AppendOne << < Ax, Ay >> > (MatA_d, MatA1_d, Ay, Ax); // O(1)

	// Add new column
	Ax++;

	// B = Transpose(A)
	MatrixTranspose << < Ax, Ay >> > (MatA1_d, MatB_d, Ax, Ay); // O(1)
	cudaMemcpy(MatB, MatB_d, Ax * Ay * sizeof(double), cudaMemcpyDeviceToHost);
	//printf("[At] = \n");
	//for (x = 0; x < (Ax * Ay); x++) {
		//printf("%f ", MatB[x]);
		//if (x != 0) {
			//if ((x % Ay) == (Ay - 1)) {
				//printf("\n");
			//}
		//}
	//}
	//printf("\n");

	// C = BA
	MatrixMul << <Ax, Ax >> > (MatB_d, MatA1_d, MatC_d, Ay, Ax, Ax, Ay); // O(Ay)
	MatrixMul << <Ax, Ax >> > (MatB_d, MatA1_d, MatC2_d, Ay, Ax, Ax, Ay);
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

	// Regularization C = C - lambda*I

	AddLambdaToDiagonal << <Ax, Ax >> > (MatC_d, lambda, Ax, Ax); // O(1)

	// Invert C
	MatrixAppendIdentity << <Ax, 2 * Ax >> > (MatC_d, MatD_d, Ax, Ax); // O(1)
	MatrixInverse << <1, 2 * Ax >> > (MatD_d, Ax, 2 * Ax); // O(Ax)
	ExtractInverse << <Ax, 2 * Ax >> > (MatD_d, MatC_d, Ax, Ax); // O(1)
	cudaMemcpy(MatC, MatC_d, Ax * Ax * sizeof(double), cudaMemcpyDeviceToHost);
	printf(" ([At][A])^-1 = \n");
	for (x = 0; x < (Ax * Ax); x++) {
		printf("%f ", MatC[x]);
		if (x != 0) {
			if ((x % (Ax)) == ((Ax)-1)) {
				printf("\n");
			}
		}
	}
	printf("\n");

	// check
	MatrixMul << <Ax, Ax >> > (MatC_d, MatC2_d, MatC3_d, Ax, Ax, Ax, Ax); // O(Ax)
	cudaMemcpy(MatC2, MatC3_d, Ax * Ax * sizeof(double), cudaMemcpyDeviceToHost);
	printf(" should be identity = \n");
	for (x = 0; x < (Ax * Ax); x++) {
		printf("%f ", MatC2[x]);
		if (x != 0) {
			if ((x % (Ax)) == ((Ax)-1)) {
				printf("\n");
			}
		}
	}
	printf("\n");

	// A = CB
	MatrixMul << <Ax, Ay >> > (MatC_d, MatB_d, MatA1_d, Ax, Ax, Ay, Ax); // O(Ax)
	cudaMemcpy(MatA, MatA_d, Ax * Ay * sizeof(double), cudaMemcpyDeviceToHost);
	//printf("Inverse * T = \n");
	//for (x = 0; x < (Ax * Ay); x++) {
		//printf("%f ", MatA[x]);
		//if (x != 0) {
			//if ((x % Ay) == (Ay - 1)) {
				//printf("\n");
			//}
		//}
	//}
	//printf("\n");

	// Beta = AE
	// E is the known vector
	MatrixMul << <1, Ax >> > (MatA1_d, MatE_d, Beta_d, Ay, Ax, 1, Ay); // O(Ay)

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
// O(Ax) time
// O(Ax * Ay) work
void linreg(double * A, double * B, double * C, int Ax, int Ay) {
	double * MatA = (double *)malloc(Ax * Ay * sizeof(double));
	double * MatA1 = (double *)malloc((Ax + 1) * Ay * sizeof(double));
	double * MatB = (double *)malloc((Ax + 1) * sizeof(double));
	double * MatC = (double *)malloc(Ay * sizeof(double));
	double * MatA_d;
	double * MatA1_d;
	double * MatB_d;
	int x;
	double * MatC_d;
	cudaMalloc((void **)&MatA_d, Ax * Ay * sizeof(double));
	cudaMalloc((void **)&MatA1_d, (Ax + 1) * Ay * sizeof(double));
	cudaMalloc((void **)&MatB_d, (Ax + 1) * sizeof(double));
	cudaMalloc((void **)&MatC_d, Ay * sizeof(double));
	cudaMemcpy(MatA_d, A, Ax * Ay * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(MatB_d, B, (Ax + 1) * sizeof(double), cudaMemcpyHostToDevice);

	// Append 1s to A
	AppendOne << <Ax, Ay >> > (MatA_d, MatA1_d, Ay, Ax); // O(1)

	// Add a column
	Ax++;

	// C = AB
	MatrixMul <<<1, Ay >> > (MatA1_d, MatB_d, MatC_d, Ax, Ay, 1, Ax); // O(Ax)

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

int main()
{
	int Asize = AX * AY * sizeof(double);
	int Bsize = BX * BY * sizeof(double);
	int Csize = (AX + 1) * sizeof(double);
	int AarrSize = AX * AY;
	int BarrSize = BX * BY;
	int CarrSize = AX + 1;
	double * MatA = (double *)malloc(Asize);
	double * MatB = (double *)malloc(Bsize);
	double * MatC = (double *)malloc(Csize);
	double * MatD = (double *)malloc(test_size * AX * sizeof(double));
	double * MatE = (double *)malloc(test_size * sizeof(double));
	double * MatA_d;
	double * MatB_d;
	double * MatC_d;
	double * MatD_d;

	memcpy(MatA, houses_m,  Asize);
	memcpy(MatB, prices_m, Bsize);
	memcpy(MatD, test_houses_m, test_size * AX * sizeof(double));

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

	// print known value vector
	printf("[B] = \n");
	for (x = 0; x < BarrSize; x++) {
		printf("%f ", MatB[x]);
		printf("\n");
	}
	printf("\n");

	// get the beta vector
	get_beta(MatA, MatB, MatC, AX, AY, 0.0655);
	printf("Beta = \n");
	for (x = 0; x < (AX + 1); x++) {
		printf("%f\n", MatC[x]);
	}
	printf("\n");

	// apply the beta vector to the input data
	linreg(MatD, MatC, MatE, AX, test_size);
	int to_add;
	int add_index;
	int out_of_range = 0;
	int error_hist[60];
	for (x = 0; x < 60; x++) {
		error_hist[x] = 0;
	}
	double sum = 0;
	double sum_s = 0;
	printf("Test Results = \n");
	for (x = 0; x < test_size; x++) {
		printf("%f vs %f\n", MatE[x], real_prices_m[x]);
		if (MatE[x] > real_prices_m[x]) {
			to_add = (MatE[x] - real_prices_m[x]);
			add_index = (int)(to_add / 10000);
			if (add_index >= 60) {
				out_of_range++;
			}
			else {
				error_hist[add_index]++;
			}
			sum += to_add;
			sum_s += (MatE[x] - real_prices_m[x]) * (MatE[x] - real_prices_m[x]);
		}
		else {
			to_add = (real_prices_m[x] - MatE[x]);
			add_index = (int)(to_add / 10000);
			if (add_index >= 60) {
				out_of_range++;
			}
			else {
				error_hist[add_index]++;
			}
			sum += to_add;
			sum_s += (real_prices_m[x] - MatE[x]) * (real_prices_m[x] - MatE[x]);
		}
	}
	printf("\n");

	sum = sum / test_size;
	sum_s = sum_s / test_size;

	for (x = 0; x < 60; x++) {
		printf("Errors between %d0000 and %d0000: %d\n", x, x + 1, error_hist[x]);
	}
	printf("Errors over 600000: %d\n", out_of_range);
	printf("\n");

	printf("Average Error = %f\n\n", sum, sum_s);

	// wait for input to close
	printf("End of test. Press Enter to close...\n");
	getchar();

	// free resources
	free(MatA);
	free(MatB);
	free(MatC);
	free(MatD);
	free(MatE);

    return 0;
}

