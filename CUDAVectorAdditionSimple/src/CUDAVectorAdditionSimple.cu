#include <iostream>
#include <cuda_runtime.h>
using std::cout;
using std::endl;

#define TYPE float
#define BLOCKSIZE 32	// Workgroup size

// kernel
__global__ void add_kernel(TYPE* A, TYPE* B, TYPE* C) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    C[n] = A[n]+B[n];
}

int main()
{
	// Sizes
	const unsigned int N = 1024u;
	dim3 block(BLOCKSIZE);		// Workgroup size
	dim3 grid(N/BLOCKSIZE);		// No. of workgroups = Global threads/workgroup size

	// Host and Device pointers;
	TYPE *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

	// Host memory allocation
	h_A = new TYPE[N];
	h_B = new TYPE[N];
	h_C = new TYPE[N];

	// host data initialisation
    for (unsigned int i = 0; i<N; i++)
    {
		h_A[i] = 3.f;
		h_B[i] = 2.f;
		h_C[i] = 1.f;
    }

	// Allocate device memory
    cudaMalloc((void**)&d_A, N*sizeof(TYPE));
    cudaMalloc((void**)&d_B, N*sizeof(TYPE));
    cudaMalloc((void**)&d_C, N*sizeof(TYPE));

    // Copy host A and B arrays to GPU
    cudaMemcpy(d_A, h_A, N*sizeof(TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*sizeof(TYPE), cudaMemcpyHostToDevice);

	cout << "Value of C[0] before kernel launch: " << h_C[0] << endl;

    // Launch kernel
    add_kernel <<<grid, block>>> (d_A, d_B, d_C);

	// Copy device C to host C
    cudaMemcpy(h_C, d_C, N*sizeof(TYPE), cudaMemcpyDeviceToHost);

	cout << "Value of C[0] after kernel execution: " << h_C[0] << endl;

    // Cleanup
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
