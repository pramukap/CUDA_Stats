#include    <iostream>
#include    <cstddef>

extern "C" {
void    print_matrix(double *X, size_t m, size_t n) 
{
    for (int i = 0; i < m; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            std::cout << *X << " ";
            X++;
        }
        std::cout << std::endl;
    }
}
}