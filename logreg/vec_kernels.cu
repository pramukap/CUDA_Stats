#include    "cuda_runtime.h"
#include    "vec_kernels.cuh"

#include    "stddef.h"
#include    <cmath>

__global__
void    mat_transpose(double *X, double *Xt, size_t m, size_t n)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= m*n)
        return;

    size_t row = gid / n;
    size_t col = gid % n;
    Xt[col * m + row] = X[row * n + col];
}

__global__
void	vec_dot_mat(double *X, double *y, double *out, size_t m, size_t n)
{
	size_t row_idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (row_idx >= m)
		return;

	out[row_idx] = 0.0;
	double accum = 0.0;
	for (size_t i = 0; i < n; i++) {
		accum += X[row_idx * n + i] * y[i];
	}
	out[row_idx] = accum;
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

__global__
void    vec_sigmoid(double *a, double *out, size_t stride, size_t n) 
{
    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x + tid;
    size_t idx = gid * stride;

    if (idx < n)
        out[idx] = 1 / (1 + exp(-a[idx]));
}

__global__
void    vec_logloss(double *h, double *y, double *out, size_t stride, size_t n) 
{
    extern __shared__ double temp[];
    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x + tid;
    size_t idx = gid * stride;

    if (idx < n) {
        temp[tid] = -(y[idx]) * log(h[idx]) - (1 - (y[idx])) * log(1 - (h[idx]));
    } else {
        temp[tid] = 0;
    }
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
void    vec_dot_asym(double *a, double *b, double *out, size_t a_stride, size_t b_stride, size_t a_n, size_t b_n)
{
    extern __shared__ double temp[];
    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x + tid;
    size_t a_idx = tid * a_stride;
    size_t b_idx = tid * b_stride;

    if (a_idx < a_n && b_idx < b_n)
        temp[tid] = a[a_idx] * b[b_idx];
    else
        temp[tid] = 0;

    __syncthreads();

    for (size_t shf = blockDim.x / 2; shf > 0; shf >>= 1) {
        if (tid < shf)
            temp[tid] += temp[tid + shf];

        __syncthreads();
    }

    if (tid == 0)
        out[blockIdx.x] = temp[0];
}
