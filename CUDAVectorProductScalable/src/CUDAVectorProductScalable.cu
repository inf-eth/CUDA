#define TYPE float
#define I 1024
#define J 1024
#define K 16
#define SIZE I*J*K
#define BLKI 32
#define BLKJ 32
#define BLKK 1
#define A(i,j,k) A[i+I*j+I*J*k]
#define B(i,j,k) B[i+I*j+I*J*k]
#define C(i,j,k) C[i+I*j+I*J*k]
#define h_A(i,j,k) h_A[i+I*j+I*J*k]
#define h_B(i,j,k) h_B[i+I*j+I*J*k]
#define h_C(i,j,k) h_C[i+I*j+I*J*k]
#define h_CR(i,j,k) h_CR[i+I*j+I*J*k]

#define CalculateError false
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

TYPE* d_A;
TYPE* d_B;
TYPE* d_C;
TYPE h_A[SIZE];
TYPE h_B[SIZE];
TYPE h_C[SIZE];
TYPE h_CR[SIZE];
dim3 block(BLKI, BLKJ, BLKK);
dim3 grid(I / BLKI, J / BLKJ, K / BLKK);

void dotProductCPU(TYPE* A, TYPE* B, TYPE* C)
{
    for (unsigned int i = 0; i < I; i++)
    {
        for (unsigned int j = 0; j < J; j++)
        {
            for (unsigned int k = 0; k < K; k++)
                C(i, j, k) = A(i, j, k) * B(i, j, k);
        }
    }
}
__global__ void dotProductGPU(TYPE* A, TYPE* B, TYPE* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    C(i, j, k) = A(i, j, k) * B(i, j, k);
}

int main() {
    auto t5 = high_resolution_clock::now();
    for (unsigned int i = 0; i < I; i++)
    {
        for (unsigned int j = 0; j < J; j++)
        {
            for (unsigned int k = 0; k < K; k++)
            {
                h_A(i, j, k) = 2 * i - 3 * j - k ^ 2;
                h_B(i, j, k) = 3 * i ^ 3 - 2 * j - 4 * k;
            }
        }
    }
    auto t1 = high_resolution_clock::now();
    dotProductCPU(h_A, h_B, h_C);
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_doubleC = t2 - t1;
    std::cout << "CPU time: " << ms_doubleC.count() << "ms\n";

    cudaMalloc((void**)&d_A, SIZE * sizeof(TYPE));
    cudaMalloc((void**)&d_B, SIZE * sizeof(TYPE));
    cudaMalloc((void**)&d_C, SIZE * sizeof(TYPE));

    // Allocate host and device memory
    cudaMemcpy(d_A, h_A, SIZE * sizeof(TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE * sizeof(TYPE), cudaMemcpyHostToDevice);

    // Launch kernel
    auto t3 = high_resolution_clock::now();
    dotProductGPU << <grid, block >> > (d_A, d_B, d_C);
    auto t4 = high_resolution_clock::now();
    duration<double, std::milli> ms_doubleG = t4 - t3;
    std::cout << "GPU time: " << ms_doubleG.count() << "ms\n";

    cudaMemcpy(h_CR, d_C, SIZE * sizeof(TYPE), cudaMemcpyDeviceToHost);

    if (CalculateError)
    {
        // Error
        TYPE Error = 0;
        for (unsigned int i = 0; i < I; i++)
        {
            for (unsigned int j = 0; j < J; j++)
            {
                for (unsigned int k = 0; k < K; k++)
                {
                    Error = Error + h_C(i, j, k) - h_CR(i, j, k);
                }
            }
        }
        std::cout << "Error is: " << Error << std::endl;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    auto t6 = high_resolution_clock::now();
    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_doubleT = t6 - t5;
    std::cout << "Total time: " << ms_doubleT.count() << "ms\n";
    return 0;
}