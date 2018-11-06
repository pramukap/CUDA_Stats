#ifndef     CUDA_STATS_VEC_KERNELS_CUH_
#define     CUDA_STATS_VEC_KERNELS_CUH_
#include    "cuda_runtime.h"

#include    <cstddef>

__device__
void    vec_dot_product(double *a, double *b, double *out, size_t stride, size_t n);

__device__
void    vec_scalar_mul(double *a, double *out, double c, size_t stride, size_t n)

#endif      // CUDA_STATS_VEC_KERNELS_CUH_
