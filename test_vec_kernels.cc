#include    "cuda_runtime.h"
#include    "vec_kernels.cuh"

#include    <cassert>
#include    <cmath>
#include    <iostream>

int     main(void) 
{
    std::cout << "Testing vec_dot_product" << std::endl;
    
    double *a, *b, *out;
    cudaMallocManaged(&a, 1024 * sizeof(double));
    cudaMallocManaged(&b, 1024 * sizeof(double));
    cudaMallocManaged(&out, sizeof(double));
    for (size_t i = 0; i < 1024; i++) {
        a[i] = b[i] = 1.0;
    }

    vec_dot_product<<<1024, 1>>>(a, b, out, 1, 1024);
    cudaDeviceSynchronize();
    assert(abs(out - 1024.0) <= 1.0);

    return 0;
}
