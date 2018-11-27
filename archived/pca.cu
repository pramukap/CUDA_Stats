#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "vec_kernels.cuh"
#include "matrixFunctions.cuh"

#include <math.h>
#include <time.h>
#include <stdio.h>

//Ax is the number of rows
//Ay is the number of cols
__global__ void meanOfFeatures(double *A, double *means, int Ax, int Ay) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	means[x] = 0;	
	
	int i;
	for (i = 0; i < Ax; i++) {
		means[x] = means[x] + A[x + (i*Ay)];
	}

	means[x] = means[x] / (double)Ax;
}

//allocate one thread for each element of Matrix A
__global__ void subMeansFromCols(double *A, double *B, double *means, int Ax, int Ay) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	int col = x % Ay;

	B[x] = A[x] - means[col];	
}

//threads allotted is equal to size of covariance matrix
//which is Ay x Ay
__global__ void calcCovMatrix(double *A, double *cov, int Ax, int Ay) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	//feature 1 and feature 2 to correlate
	//corresponds to row and col in covariance matrix
	int f1 = x / Ay;
	int f2 = x % Ay;

	int i;
	for (i = 0; i < Ax; i++) {
		cov[x] += (A[f1 + (i * Ay)] * A[f2 + (i * Ay)]);
	}

	cov[x] /= (Ax - 1);
}	

//https://en.wikipedia.org/wiki/Principal_component_analysis#Covariance-free_computation
//random vector that is of length num_features
__global__ void genRandVector(double *r, curandState *state, unsigned long clock_seed) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	unsigned long seed = 132241684309342543 + clock_seed;
	curand_init (seed, x, 0, &state[x]); 
	
	curandState localState = state[x];
	r[x] = (double)curand_uniform(&localState);
	state[x] = localState;
}

__global__ void clearSVector(double *s, int n) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	s[x] = 0;
}

//allocate a thread for each element of row
__global__ void getRow(double *X, double *out, int row_index, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	out[x] = X[x + (row_index * cols)];
	//out[x] = row_index;
}

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

__global__
void    vec_add(double *a, double *b, double *out, size_t stride, size_t n)
{
    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x + tid;
    size_t idx = gid * stride;

    if (idx < n)
        out[idx] = a[idx] + b[idx];
}

__global__
void    vec_dot_product(double *a, double *b, double *out, size_t stride, size_t n)
{
    extern __shared__ double temp[];
    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x + tid;
    size_t idx = gid * stride;

    temp[tid] = (idx < n) ? a[idx] * b[idx] : 0;
    __syncthreads();

    for (size_t shf = blockDim.x / 2; shf > 0; shf >>= 1) {
        if (tid < shf) {
            temp[tid] += temp[tid + shf];  
        }

        __syncthreads();
    }

    if (tid == 0)
        out[blockIdx.x] = temp[0];
}

__global__
void    vec_scalar_mul(double *a, double *out, double c, size_t stride, size_t n) 
{
    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x + tid;
    size_t idx = gid * stride;

    if (idx < n)
        out[idx] = a[idx] * c;
}

double *pca(double *data, int rows, int cols, int iterations) {
	//host
	volatile int i, j, k;
	double dot_prod = 0;
	double *r = (double *)calloc(cols, sizeof(double));		
	double *s = (double *)calloc(cols, sizeof(double));		
	double *eigenvalues = (double *)malloc(cols * sizeof(double));
	unsigned long clock_seed = (unsigned long)clock();

	//device
	double *data_d, *means_d, *zero_mean_data_d, *r_d, *r_unit_d, *s_d, *x_row_d, *r_x_dot_d, *add_to_s_d, *eigenvalues_d;
	curandState *state_d;

	cudaMalloc((void**)&data_d, rows * cols * sizeof(double));
	cudaMalloc((void**)&means_d, cols * sizeof(double));
	cudaMalloc((void**)&zero_mean_data_d, rows * cols * sizeof(double));
	cudaMalloc((void**)&r_d, cols * sizeof(double));
	cudaMalloc((void**)&r_unit_d, cols * sizeof(double));
	cudaMalloc((void**)&s_d, cols * sizeof(double));
	cudaMalloc((void**)&x_row_d, cols * sizeof(double));
	cudaMalloc((void**)&r_x_dot_d, cols * sizeof(double));
	cudaMalloc((void**)&add_to_s_d, cols * sizeof(double));
	cudaMalloc((void**)&eigenvalues_d, cols * sizeof(double));
	cudaMalloc((void**)&state_d, cols * sizeof(curandState));

	printf("Data:\n");
	for (k = 0; k < 6; k++) {
		printf("%f ", data[k]);
		if (k % 3 == 2) {
			printf("\n");
		}
	}
	printf("\n\n");
	
	cudaMemcpy(r_d, r, cols * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(data_d, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);

	meanOfFeatures <<<1, cols>>> (data_d, means_d, rows, cols);
	
    cudaMemcpy(r, means_d, cols * sizeof(double), cudaMemcpyDeviceToHost);

	printf("Means:\n");
	for (k = 0; k < 3; k++) {
		printf("%f ", r[k]);
	}
	printf("\n\n");

	subMeansFromCols <<<rows, cols>>> (data_d, zero_mean_data_d, means_d, rows, cols);

    cudaMemcpy(data, zero_mean_data_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
	printf("Zero Mean Data:\n");
	for (k = 0; k < 6; k++) {
		printf("%f ", data[k]);
		if (k % 3 == 2) {
			printf("\n");
		}
	}
	printf("\n\n");

	genRandVector <<<1, cols>>> (r_d, state_d, clock_seed);

	cudaDeviceSynchronize();
    cudaMemcpy(r, r_d, cols * sizeof(double), cudaMemcpyDeviceToHost);

	printf("Random Values:\n");
	for (k = 0; k < 3; k++) {
		printf("%f ", r[k]);
	}
	printf("\n\n");
	
	double r_len = 0;
	for (i = 0; i < cols; i++) {
		r_len += (r[i] * r[i]);	
	}

	r_len = 1/sqrt(r_len);
	printf("Rlen: %f\n\n", r_len);
	
	vec_scalar_mul <<<1, cols>>> (r_d, r_unit_d, r_len, 1, cols); 
	
	cudaDeviceSynchronize();
    cudaMemcpy(r, r_unit_d, cols * sizeof(double), cudaMemcpyDeviceToHost);

	printf("Random Unit Vector:\n");
	for (k = 0; k < 3; k++) {
		printf("%f ", r[k]);
	}
	printf("\n\n");

	double s_len;
	for (i = 0; i < iterations; i++) {
		printf("Iteration: %i\n", i);
		//set s to 0
		clearSVector <<<1, cols>>> (s_d, cols);

		for (j = 0; j < rows; j++) {
			printf("Row: %i\n", j);
			cudaDeviceSynchronize();
			getRow <<<1, cols>>> (zero_mean_data_d, x_row_d, j, cols);

			cudaDeviceSynchronize();
    		cudaMemcpy(s, x_row_d, cols * sizeof(double), cudaMemcpyDeviceToHost);

			printf("Row:\n");
			for (k = 0; k < 3; k++) {
				printf("%f ", s[k]);
			}
			printf("\n\n");

			//printf("Got row\n");
			//output is at index 0 of data_r_d 
			vec_dot_product <<<1, cols>>> (x_row_d, r_unit_d, r_x_dot_d, 1, cols);
			//printf("Got dot product\n");

			cudaDeviceSynchronize();
    		cudaMemcpy(&dot_prod, r_x_dot_d, sizeof(double), cudaMemcpyDeviceToHost);
			//printf("Isolated scalar value");

			vec_scalar_mul <<<1, cols>>> (x_row_d, add_to_s_d, dot_prod, 1, cols); 
			//printf("Did scalar mult\n");

			vec_add <<<1, cols>>> (s_d, add_to_s_d, s_d, 1, cols);
			//printf("Did vector add\n");
		}
		cudaDeviceSynchronize();
    	cudaMemcpy(s, s_d, cols * sizeof(double), cudaMemcpyDeviceToHost);

		printf("S:\n");
		for (k = 0; k < 3; k++) {
			printf("%f ", s[k]);
		}
		printf("\n");

		
		MatrixMul <<<1, cols>>> (r_unit_d, s_d, eigenvalues_d, cols, 1, 1, cols);

		cudaDeviceSynchronize();
    	cudaMemcpy(eigenvalues, eigenvalues_d, cols * sizeof(double), cudaMemcpyDeviceToHost);

		printf("Eigenvalues:\n");
		for (k = 0; k < 3; k++) {
			printf("%f ", eigenvalues[k]);
		}
		printf("\n");

		//printf("Did matrix mult\n");

		cudaDeviceSynchronize();
    	cudaMemcpy(s, s_d, cols * sizeof(double), cudaMemcpyDeviceToHost);
		s_len = 0;
		for (k = 0; k < cols; k++) {
			s_len += (s[i] * s[i]);	
		}

		s_len = 1/sqrt(s_len);
	
		vec_scalar_mul <<<1, cols>>> (r_unit_d, r_unit_d, s_len, 1, cols); 
	}

	cudaDeviceSynchronize();
	printf("Finished calculating eigenvalues\n");

    cudaMemcpy(eigenvalues, eigenvalues_d, cols * sizeof(double), cudaMemcpyDeviceToHost);

	printf("Eigenvalues:\n");
	for (i = 0; i < 3; i++) {
		printf("%f ", eigenvalues[i]);
	}
	printf("\n");

	free(r);
	free(s);
	cudaFree(data_d);
	cudaFree(means_d);
	cudaFree(zero_mean_data_d);
	cudaFree(r_d);
	cudaFree(r_unit_d);
	cudaFree(s_d);
	cudaFree(x_row_d);
	cudaFree(r_x_dot_d);
	cudaFree(add_to_s_d);
	cudaFree(state_d);
	
	return eigenvalues;
}

#define SIZE 5

int main() {
	double data[6] = {1, 2, 3, 4, 5, 6};
	double *eigenvalues = pca(data, 2, 3, 2);

	int i;
	for (i = 0; i < 3; i++) {
		printf("%f ", eigenvalues[i]);
	}
	printf("\n");
	/*
	double r[5] = {0, 0, 0, 0, 0};		
	unsigned long clock_seed = (unsigned long)clock();

	double *r_d;
	curandState *state_d;
	unsigned long *clock_seed_d;

	cudaMalloc((void**)&r_d, SIZE * sizeof(double));
	cudaMalloc((void**)&state_d, SIZE * sizeof(curandState));
	cudaMalloc((void**)&clock_seed_d, sizeof(unsigned long));

	cudaMemcpy(r_d, r, SIZE * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(clock_seed_d, &clock_seed, sizeof(unsigned long), cudaMemcpyHostToDevice);

	genRandVector<<<5, 1>>>(r_d, state_d, (double)clock());

    cudaMemcpy(r, r_d, SIZE * sizeof(double), cudaMemcpyDeviceToHost);

	int i;
	for (i = 0; i < SIZE; i++) {
		printf("%f ", r[i]);
	}
	printf("\n");
	*/
}

			/*
		vec_dot_mat(zero_mean_data_d, r_unit_d, data_r_d, rows, cols)

		MatrixMul(zero_mean_data_d, data_r_d, add_to_s, cols, rows, 1, cols);
		vec_add(s_d, add_to_s, new_s_d, 1, cols);
		
		MatrixMul(r_unit_d, , double * C, int Ax, int Ay, int Bx, int By)
				*/
