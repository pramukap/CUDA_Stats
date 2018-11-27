#include    "cuda_runtime.h"
#include    "vec_kernels.cuh"

#include    <cassert>
#include    <cmath>
#include    <iostream>

int     main(void) 
{
    std::cout << "Test: vec_dot_product" << std::endl;
    
    double *a, *b, *out;
    cudaMallocManaged(&a, 1024 * sizeof(double));
    cudaMallocManaged(&b, 1024 * sizeof(double));
    cudaMallocManaged(&out, sizeof(double));
    for (size_t i = 0; i < 1024; i++) {
        a[i] = 1.0;
        b[i] = 1.0;
    }

    vec_dot_product<<<1, 1024, sizeof(double) * 1024>>>(a, b, out, 1, 1024);
    cudaDeviceSynchronize();
    assert(abs(*out - 1024.0) <= 1.0);

    cudaFree(a);
    cudaFree(b);
    cudaFree(out);
    std::cout << "Passed: vec_dot_product" << std::endl;
    std::cout << "Test: vec_scalar_mul" << std::endl;

    cudaMallocManaged(&a, 1024 * sizeof(double));
    cudaMallocManaged(&b, 1024 * sizeof(double));
    
    for (size_t i = 0; i < 1024; i++) {
        a[i] = b[i] = 1.0;
    }

    vec_scalar_mul<<<1, 1024>>>(a, b, 1024, 1, 1024);
    cudaDeviceSynchronize();
    for (size_t i = 0; i < 1024; i++) {
        assert(abs(b[i] - 1024.0) <= 1.0);
    }
    std::cout << "Passed: vec_scalar_mul" << std::endl;

    cudaFree(a);
    cudaFree(b);
    return 0;
}
