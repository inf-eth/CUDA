#define TYPE float
#define SIZE 1024*1024
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

void dotProductCPU(TYPE* A, TYPE* B, TYPE* C)
{
    for (unsigned int i = 0; i < SIZE; i++)
        C[i] = A[i] * B[i];
}
__global__ void dotProductGPU(TYPE* A, TYPE* B, TYPE* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    C[idx] = A[idx] * B[idx];
}

int main() {
    auto t5 = high_resolution_clock::now();
    for (unsigned int i = 0; i < SIZE; i++)
    {
        h_A[i] = 2 * i - 3 * i - i ^ 2;
        h_B[i] = 3 * i ^ 3 - 2 * i - 4 * i;
    }
    auto t1 = high_resolution_clock::now();
    dotProductCPU(h_A, h_B, h_C);
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_doubleC = t2 - t1;
    std::cout << "CPU time: " << ms_doubleC.count() << "ms\n";

    d_A = (TYPE*)malloc(SIZE * sizeof(TYPE));
    d_B = (TYPE*)malloc(SIZE * sizeof(TYPE));
    d_C = (TYPE*)malloc(SIZE * sizeof(TYPE));

    // Allocate host and device memory
    cudaMemcpy(d_A, h_A, SIZE * sizeof(TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE * sizeof(TYPE), cudaMemcpyHostToDevice);

    // Launch kernel
    auto t3 = high_resolution_clock::now();
    dotProductGPU << <1, SIZE >> > (d_A, d_B, d_C);
    auto t4 = high_resolution_clock::now();
    duration<double, std::milli> ms_doubleG = t4 - t3;
    std::cout << "GPU time: " << ms_doubleG.count() << "ms\n";

    cudaMemcpy(h_CR, d_C, SIZE * sizeof(TYPE), cudaMemcpyDeviceToHost);

    // Error
    TYPE Error = 0;
    for (unsigned int i = 0; i < SIZE; i++)
        Error = Error + h_C[i] - h_CR[i];

    std::cout << "Error is: " << Error << std::endl;

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