
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "house_med.h"
#include "time.h"
#include "math.h"

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
		}

		__syncthreads();
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
	AppendOne << < Ax, Ay >> > (MatA_d, MatA1_d, Ay, Ax); // O(1)

	// Add new column
	Ax++;

	// B = Transpose(A)
	MatrixTranspose << < Ax, Ay >> > (MatA1_d, MatB_d, Ax, Ay); // O(1)

	// C = BA
	MatrixMul << <Ax, Ax >> > (MatB_d, MatA1_d, MatC_d, Ay, Ax, Ax, Ay); // O(Ay)

	// Regularization C = C - lambda*I
	AddLambdaToDiagonal << <Ax, Ax >> > (MatC_d, lambda, Ax, Ax); // O(1)

	// Invert C
	MatrixAppendIdentity << <Ax, 2 * Ax >> > (MatC_d, MatD_d, Ax, Ax); // O(1)
	MatrixInverse << <1, 2 * Ax >> > (MatD_d, Ax, 2 * Ax); // O(Ax)
	ExtractInverse << <Ax, 2 * Ax >> > (MatD_d, MatC_d, Ax, Ax); // O(1)

	// A = CB
	MatrixMul << <Ax, Ay >> > (MatC_d, MatB_d, MatA1_d, Ax, Ax, Ay, Ax); // O(Ax)

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
	// Training data arrays
	int Asize = AX * AY * sizeof(double);
	int A2size = AX * (AY / 10) * sizeof(double);
	int A3size = AX * (AY / 100) * sizeof(double);
	int AarrSize = AX * AY;
	double * MatA = (double *)malloc(Asize);
	double * MatA2 = (double *)malloc(A2size);
	double * MatA3 = (double *)malloc(A3size);
	memcpy(MatA, houses_m, Asize);
	memcpy(MatA2, houses_m, A2size);
	memcpy(MatA3, houses_m, A3size);

	// Known price arrays
	int Bsize = BX * BY * sizeof(double);
	int B2size = BX * (BY / 10) * sizeof(double);
	int B3size = BX * (BY / 100) * sizeof(double);
	int BarrSize = BX * BY;
	double * MatB = (double *)malloc(Bsize);
	double * MatB2 = (double *)malloc(Bsize);
	double * MatB3 = (double *)malloc(Bsize);
	memcpy(MatB, prices_m, Bsize);
	memcpy(MatB2, prices_m, B2size);
	memcpy(MatB3, prices_m, B3size);

	// Output Arrays
	int Csize = (AX + 1) * sizeof(double);
	int CarrSize = AX + 1;
	double * MatC = (double *)malloc(Csize);
	double * MatD = (double *)malloc(test_size * AX * sizeof(double));
	double * MatE = (double *)malloc(test_size * sizeof(double));
	double * MatE2 = (double *)malloc(test_size * sizeof(double));
	double * MatE3 = (double *)malloc(test_size * sizeof(double));
	memcpy(MatD, test_houses_m, test_size * AX * sizeof(double));
	
	// Set up timing variables
	clock_t start, end;
	double time3, time2, time;

	int x;

	// Test with 10 training observations
	// Fit a line to the training data
	start = clock();
	get_beta(MatA3, MatB3, MatC, AX, (AY / 100), 0.0655);
	end = clock();
	time3 = ((double)(end - start)) / CLOCKS_PER_SEC;
	// Apply the beta vector to the input data to get the predicted values
	linreg(MatD, MatC, MatE3, AX, test_size);
	
	// Test with 100 training observations
	start = clock();
	get_beta(MatA2, MatB2, MatC, AX, (AY / 10), 0.0655);
	end = clock();
	time2 = ((double)(end - start)) / CLOCKS_PER_SEC;
	linreg(MatD, MatC, MatE2, AX, test_size);

	// Test with 1000 training observations
	start = clock();
	get_beta(MatA, MatB, MatC, AX, AY, 0.0655);
	end = clock();
	time = ((double)(end - start)) / CLOCKS_PER_SEC;
	linreg(MatD, MatC, MatE, AX, test_size);

	// Print test reports
	double to_add, to_add2, to_add3;
	int add_index;
	int out_of_range = 0;
	int error_hist[60];
	for (x = 0; x < 60; x++) {
		error_hist[x] = 0;
	}
	double sum = 0;
	double sum2 = 0;
	double sum3 = 0;
	double sum_s = 0;
	double sum_s2 = 0;
	double sum_s3 = 0;

	// Calculate errors
	for (x = 0; x < test_size; x++) {
		// get the error
		to_add = (MatE[x] - real_prices_m[x]);
		to_add2 = (MatE2[x] - real_prices_m[x]);
		to_add3 = (MatE3[x] - real_prices_m[x]);

		// absolute value
		if (to_add < 0) {
			to_add = to_add * -1;
		}
		if (to_add2 < 0) {
			to_add2 = to_add2 * -1;
		}
		if (to_add3 < 0) {
			to_add3 = to_add3 * -1;
		}

		// update histogram for 1000 element test
		add_index = (int)(to_add / 10000);
		if (add_index >= 60) {
			out_of_range++;
		}
		else {
			error_hist[add_index]++;
		}

		// update the sum and sum of squares
		sum += to_add;
		sum2 += to_add2;
		sum3 += to_add3;
		sum_s += (to_add * to_add);
		sum_s2 += (to_add2 * to_add2);
		sum_s3 += (to_add3 * to_add3);
	}
	// calculate average error
	sum = sum / test_size;
	sum2 = sum2 / test_size;
	sum3 = sum3 / test_size;
	// calculate RMSE
	sum_s = sqrt(sum_s / test_size);
	sum_s2 = sqrt(sum_s2 / test_size);
	sum_s3 = sqrt(sum_s3 / test_size);

	// Print 10 element test report
	printf("Results for 10 element test:\n\n");
	for (x = 0; x < test_size; x++) {
		if (x % 2 == 0 && x != 0) {
			printf("\n");
			printf("Predicted: %f  \tActual: %f\t\t", MatE3[x], real_prices_m[x]);
		}
		else {
			printf("Predicted: %f  \tActual: %f\t\t", MatE3[x], real_prices_m[x]);
		}
	}
	printf("\n");
	printf("Best fit calculation time: %f\n", time3);
	printf("Average error: %f\n", sum3);
	printf("RMSE: %f\n\n\n", sum_s3);

	// Print 100 element test report
	printf("Results for 100 element test:\n\n");
	for (x = 0; x < test_size; x++) {
		if (x % 2 == 0 && x != 0) {
			printf("\n");
			printf("Predicted: %f  \tActual: %f\t\t", MatE2[x], real_prices_m[x]);
		}
		else {
			printf("Predicted: %f  \tActual: %f\t\t", MatE2[x], real_prices_m[x]);
		}
	}
	printf("\n");
	printf("Best fit calculation time: %f\n", time2);
	printf("Average error: %f\n", sum2);
	printf("RMSE: %f\n\n\n", sum_s2);

	// Print 1000 element test report
	printf("Results for 1000 element test:\n\n");
	for (x = 0; x < test_size; x++) {
		if (x % 2 == 0 && x != 0) {
			printf("\n");
			printf("Predicted: %f  \tActual: %f\t\t", MatE[x], real_prices_m[x]);
		}
		else {
			printf("Predicted: %f  \tActual: %f\t\t", MatE[x], real_prices_m[x]);
		}
	}
	printf("\n");
	printf("Best fit calculation time: %f\n", time);
	printf("Average error: %f\n", sum);
	printf("RMSE: %f\n\n\n", sum_s);

	//for (x = 0; x < 60; x++) {
		//printf("Errors between %d0000 and %d0000: %d\n", x, x + 1, error_hist[x]);
	//}
	//printf("Errors over 600000: %d\n", out_of_range);
	//printf("\n");

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

