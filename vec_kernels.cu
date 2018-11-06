#include    "cuda_runtime.h"
#include    "vec_kernels.cuh"

#include    <cstddef>

__device__
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

__device__
void    vec_scalar_mul(double *a, double *out, double c, size_t stride, size_t n) 
{
    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x + tid;
    size_t idx = gid * stride;

    if (idx < n)
        out[idx] = a[idx] * c;
}
