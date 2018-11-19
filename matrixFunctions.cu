#include "cuda_runtime.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "time.h"
#include "math.h"

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
__global__ void MatrixSMul(double * A, double * B, double scalar) {
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
			Bindex = (x % Bx) + Bx * count;
			prod = A[Aindex] * B[Bindex];
			C[x] += prod;
		}
	}
}

// Adds the value lambda to the diagonal of the input matrix
// O(1) time
// O(Ax * Ay) work
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
	MatrixMul << <1, Ay >> > (MatA1_d, MatB_d, MatC_d, Ax, Ay, 1, Ax); // O(Ax)

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

// Training set is the data the line will be fit to
// Known values corrospond to the training set
// Test set will be used to test the line of best fit
// Test values are the actual values of the test set
// X is the number of features in the dataset.
// Y is the number of elements in the training set. should be less than 1024
// Yt is the number of elements in the test set
extern "C" {
	void linreg_test(double * training_set, double * known_values, double * test_set, double * test_values, int X, int Y, int Yt)
	{
		int AX = X;
		int AY = Y;
		int BX = 1;
		int BY = Y;

		// Training data arrays
		int Asize = AX * AY * sizeof(double);
		int A2size = AX * (AY / 10) * sizeof(double);
		int A3size = AX * (AY / 100) * sizeof(double);
		int AarrSize = AX * AY;
		double * MatA = (double *)malloc(Asize);
		double * MatA2 = (double *)malloc(A2size);
		double * MatA3 = (double *)malloc(A3size);
		memcpy(MatA, training_set, Asize);
		memcpy(MatA2, training_set, A2size);
		memcpy(MatA3, training_set, A3size);

		// Known price arrays
		int Bsize = BX * BY * sizeof(double);
		int B2size = BX * (BY / 10) * sizeof(double);
		int B3size = BX * (BY / 100) * sizeof(double);
		int BarrSize = BX * BY;
		double * MatB = (double *)malloc(Bsize);
		double * MatB2 = (double *)malloc(Bsize);
		double * MatB3 = (double *)malloc(Bsize);
		memcpy(MatB, known_values, Bsize);
		memcpy(MatB2, known_values, B2size);
		memcpy(MatB3, known_values, B3size);

		// Output Arrays
		int Csize = (AX + 1) * sizeof(double);
		int CarrSize = AX + 1;
		double * MatC = (double *)malloc(Csize);
		double * MatC2 = (double *)malloc(Csize);
		double * MatC3 = (double *)malloc(Csize);
		double * MatD = (double *)malloc(Yt * AX * sizeof(double));
		double * MatE = (double *)malloc(Yt * sizeof(double));
		double * MatE2 = (double *)malloc(Yt * sizeof(double));
		double * MatE3 = (double *)malloc(Yt * sizeof(double));
		memcpy(MatD, test_set, Yt * AX * sizeof(double));

		// Set up timing variables
		clock_t start, end;
		double time3, time2, time;

		int x;

		// Test with 10 training observations
		// Fit a line to the training data
		start = clock();
		get_beta(MatA3, MatB3, MatC3, AX, (AY / 100), 0.0655);
		end = clock();
		time3 = ((double)(end - start)) / CLOCKS_PER_SEC;
		// Apply the beta vector to the input data to get the predicted values
		linreg(MatD, MatC3, MatE3, AX, Yt);

		// Test with 100 training observations
		start = clock();
		get_beta(MatA2, MatB2, MatC2, AX, (AY / 10), 0.0655);
		end = clock();
		time2 = ((double)(end - start)) / CLOCKS_PER_SEC;
		linreg(MatD, MatC2, MatE2, AX, Yt);

		// Test with 1000 training observations
		start = clock();
		get_beta(MatA, MatB, MatC, AX, AY, 0.0655);
		end = clock();
		time = ((double)(end - start)) / CLOCKS_PER_SEC;
		linreg(MatD, MatC, MatE, AX, Yt);

		// Print test reports
		double to_add, to_add2, to_add3;
		double sum = 0;
		double sum2 = 0;
		double sum3 = 0;
		double sum_s = 0;
		double sum_s2 = 0;
		double sum_s3 = 0;

		// Calculate errors
		for (x = 0; x < Yt; x++) {
			// get the error
			to_add = (MatE[x] - test_values[x]);
			to_add2 = (MatE2[x] - test_values[x]);
			to_add3 = (MatE3[x] - test_values[x]);

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

			// update the sum and sum of squares
			sum += to_add;
			sum2 += to_add2;
			sum3 += to_add3;
			sum_s += (to_add * to_add);
			sum_s2 += (to_add2 * to_add2);
			sum_s3 += (to_add3 * to_add3);
		}
		// calculate average error
		sum = sum / Yt;
		sum2 = sum2 / Yt;
		sum3 = sum3 / Yt;
		// calculate RMSE
		sum_s = sqrt(sum_s / Yt);
		sum_s2 = sqrt(sum_s2 / Yt);
		sum_s3 = sqrt(sum_s3 / Yt);

		// Print 10 element test report
		printf("Results for %d element test:\n\n", (AY / 100));
		printf("Beta = \n");
		for (x = 0; x < (AX + 1); x++) {
			printf("%f\n", MatC3[x]);
		}
		printf("\n");
		for (x = 0; x < Yt; x++) {
			if (x % 2 == 0 && x != 0) {
				printf("\n");
				printf("Predicted: %f  \tActual: %f\t\t", MatE3[x], test_values[x]);
			}
			else {
				printf("Predicted: %f  \tActual: %f\t\t", MatE3[x], test_values[x]);
			}
		}
		printf("\n");
		printf("Best fit calculation time: %f\n", time3);
		printf("Average error: %f\n", sum3);
		printf("RMSE: %f\n\n\n", sum_s3);

		// Print 100 element test report
		printf("Results for %d element test:\n\n", (AY / 10));
		printf("Beta = \n");
		for (x = 0; x < (AX + 1); x++) {
			printf("%f\n", MatC2[x]);
		}
		printf("\n");
		for (x = 0; x < Yt; x++) {
			if (x % 2 == 0 && x != 0) {
				printf("\n");
				printf("Predicted: %f  \tActual: %f\t\t", MatE2[x], test_values[x]);
			}
			else {
				printf("Predicted: %f  \tActual: %f\t\t", MatE2[x], test_values[x]);
			}
		}
		printf("\n");
		printf("Best fit calculation time: %f\n", time2);
		printf("Average error: %f\n", sum2);
		printf("RMSE: %f\n\n\n", sum_s2);

		// Print 1000 element test report
		printf("Results for %d element test:\n\n", AY);
		printf("Beta = \n");
		for (x = 0; x < (AX + 1); x++) {
			printf("%f\n", MatC[x]);
		}
		printf("\n");
		for (x = 0; x < Yt; x++) {
			if (x % 2 == 0 && x != 0) {
				printf("\n");
				printf("Predicted: %f  \tActual: %f\t\t", MatE[x], test_values[x]);
			}
			else {
				printf("Predicted: %f  \tActual: %f\t\t", MatE[x], test_values[x]);
			}
		}
		printf("\n");
		printf("Best fit calculation time: %f\n", time);
		printf("Average error: %f\n", sum);
		printf("RMSE: %f\n\n\n", sum_s);

		// free resources
		free(MatA);
		free(MatB);
		free(MatC);
		free(MatD);
		free(MatE);

	}
}
