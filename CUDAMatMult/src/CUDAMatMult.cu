#define TYPE float
#ifdef __linux__
#include <openacc.h>
#endif
#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <stdio.h>
using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

struct msClock
{
	typedef std::chrono::high_resolution_clock clock;
	std::chrono::time_point<clock> t1, t2;
	void Start() { t1 = high_resolution_clock::now(); }
	void Stop() { t2 = high_resolution_clock::now(); }
	double ElapsedTime()
	{
		duration<double, std::milli> ms_doubleC = t2-t1;
		return ms_doubleC.count();
	}
}
Clock;

// CUDA block sizes for kernel
#define BLKI 8
#define BLKJ 8
#define BLKK 1

void NullMat(TYPE*, int, int);
TYPE diffMat(TYPE*, TYPE*, int , int);
void initialiseMat(TYPE*, int, int);
void displayMat(TYPE*, int, int);
void matMult(TYPE*, TYPE*, TYPE*, int, int, int, int);
void matMultOMP(TYPE*, TYPE*, TYPE*, int, int, int, int);
#ifdef __linux__
void matMultOACC(TYPE*, TYPE*, TYPE*, int, int, int, int);
#endif

// CUDA kernels (1D (sudo2D) and 2D)
__global__ void MatMultKernel(TYPE* C, TYPE* A, TYPE* B, int rA, int cA, int rB, int cB)
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	const int i = n/cB;
	const int j = n%cB;

	TYPE Sum = 0;
	for (int k=0; k<cA; k++)
		Sum = Sum+A[k+i*cA]*B[j+k*cB];
	C[j+i*cB] = Sum;
}
__global__ void MatMultKernel2D(TYPE* C, TYPE* A, TYPE* B, int rA, int cA, int rB, int cB)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	TYPE Sum = 0;
	for (int k=0; k<cA; k++)
		Sum = Sum+A[k+i*cA]*B[j+k*cB];
	C[j+i*cB] = Sum;

}

__global__ void MatMultKernel2DShared(TYPE* C, TYPE* A, TYPE* B, int rA, int cA, int rB, int cB)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int li = threadIdx.x;
	int lj = threadIdx.y;

	//if (li==0 && lj==0)
	//	printf("li=%d, lj=%d\n",li,lj);
	//printf("%d\n", i+j);
	__shared__ TYPE lA[BLKI][BLKJ];
	__shared__ TYPE lB[BLKI][BLKJ];

	TYPE SubSum = 0;

	int gridrows = rB/BLKI;
	//int gridcols = cA/BLKJ;

	for (int blockINo=0, blockJNo=0; blockINo<gridrows; blockINo++, blockJNo++)
	{
		int iOffset = blockIdx.x * blockDim.x;
		int jOffset = blockIdx.y * blockDim.y;

		//if (li==0 && lj==0)
		//	printf("iOffset=%d, jOffset=%d\n",iOffset,jOffset);

		int ii = iOffset+li;
		int jj = blockJNo*BLKJ+lj;
		lA[li][lj] = A[jj+ii*cA];	// A[ii][jj] = A[offset+li][

		ii = blockINo*BLKI+li;
		jj = jOffset+lj;
		lB[li][lj] = B[jj+ii*cB];	// B[ii][jj]

		__syncthreads();

		//if (li==0 && lj==0)
		//{
			//printf("li=%d, lj=%d: ii=%d, jj=%d\n",li,lj,ii,jj);
			//TYPE* temp = (TYPE*)&lA[0][0];
			//for (int _i=0;_i<BLKI*BLKJ;_i++)
			//	printf("%d ", (int)temp[_i]);
		//}

		for (int k=0; k<BLKJ; k++)
			SubSum = SubSum + lA[li][k] * lB[k][lj];

		__syncthreads();
	}
	C[j+i*cB] = SubSum;

}

int main()
{
	//vector<Device_Info> test;
	// compile OpenCL C code for the device with the given id
	//Device device(select_device_with_id(0), "MatMult_Kernels.cl");

	// size of vectors
	int rA = 1024;
	int cA = 1024;
	int rB = cA;
	int cB = 1024;
	// 1D kernel
	dim3 block(BLKI);
	dim3 grid((rA*cB)/BLKI);
	// 2D kernel
	dim3 block2D(BLKI, BLKJ);
	dim3 grid2D(rA/BLKI, cB/BLKJ);

	// allocate memory on both host and device
	TYPE* A = new TYPE[rA*cA];
	TYPE* B = new TYPE[rB*cB];
	TYPE* C = new TYPE[rA*cB];
	TYPE* gpuCResult = new TYPE[rA*cB];

	TYPE* gpuA;
	TYPE* gpuB;
	TYPE* gpuC;
	cudaMalloc((void**)&gpuA, rA*cA * sizeof(TYPE));
	cudaMalloc((void**)&gpuB, rB*cB * sizeof(TYPE));
	cudaMalloc((void**)&gpuC, rA*cB * sizeof(TYPE));

	//Memory<TYPE> A(device, rA*cA, 1);
	//Memory<TYPE> B(device, rB*cB, 1);
	//Memory<TYPE> C(device, rA*cB, 1);
	//Memory<TYPE> gpuC(device, rA*cB, 1);

	// initialise memory
	initialiseMat(A, rA, cA);
	initialiseMat(B, rB, cB);

	displayMat(A,rA,cA);
	displayMat(B,rB,cB);

	Clock.Start();
	//matMult(C,A,B,rA,cA,rB,cB);
	Clock.Stop();
	cout << "Time taken (single): " << Clock.ElapsedTime() << " ms." << endl;
	//cout <<("Time taken (single): "+to_string(Clock.ElapsedTime(), 3)+" ms.");

	displayMat(C,rA,cB);

	Clock.Start();
	//matMultOMP(C,A,B,rA,cA,rB,cB);
	Clock.Stop();
	cout << "Time taken (OMP): " << Clock.ElapsedTime() << " ms." << endl;

	#ifdef __linux__
	Clock.Start();
	//matMultOACC(C,A,B,rA,cA,rB,cB);
	Clock.Stop();
	cout << "Time taken (OACC): " << Clock.ElapsedTime() << " ms." << endl;
	#endif

	// ================================= CUDA Mat Mult =========================================
	cout << "Value before kernel execution: C[0]: " << gpuCResult[0] << endl;
	
	for ( int i=0; i<10; i++)
	{
		Clock.Start();
		// copy data from host memory to device memory
		cudaMemcpy(gpuA, A, rA*cA * sizeof(TYPE), cudaMemcpyHostToDevice);
		cudaMemcpy(gpuB, B, rB*cB * sizeof(TYPE), cudaMemcpyHostToDevice);
		// run add_kernel on the device
		//MatMultKernel <<< grid, block >>> (gpuC, gpuA, gpuB, rA, cA, rB, cB);
		MatMultKernel2D <<< grid2D, block2D >>> (gpuC, gpuA, gpuB, rA, cA, rB, cB);
		//MatMultKernel2DShared <<< grid2D, block2D >>> (gpuC, gpuA, gpuB, rA, cA, rB, cB);
		// copy data from device memory to host memory
		cudaMemcpy(gpuCResult, gpuC, rA*cB * sizeof(TYPE), cudaMemcpyDeviceToHost);
		Clock.Stop();
		cout << "Time taken (CUDA): " << Clock.ElapsedTime() << " ms." << endl;
		// =========================================================================================
	}
	cout << "========== Shared =========" << endl;
	for ( int i=0; i<10; i++)
	{
		Clock.Start();
		// copy data from host memory to device memory
		cudaMemcpy(gpuA, A, rA*cA * sizeof(TYPE), cudaMemcpyHostToDevice);
		cudaMemcpy(gpuB, B, rB*cB * sizeof(TYPE), cudaMemcpyHostToDevice);
		// run add_kernel on the device
		//MatMultKernel <<< grid, block >>> (gpuC, gpuA, gpuB, rA, cA, rB, cB);
		//MatMultKernel2D <<< grid2D, block2D >>> (gpuC, gpuA, gpuB, rA, cA, rB, cB);
		MatMultKernel2DShared <<< grid2D, block2D >>> (gpuC, gpuA, gpuB, rA, cA, rB, cB);
		// copy data from device memory to host memory
		cudaMemcpy(gpuCResult, gpuC, rA*cB * sizeof(TYPE), cudaMemcpyDeviceToHost);
		Clock.Stop();
		cout << "Time taken (CUDA): " << Clock.ElapsedTime() << " ms." << endl;
		// =========================================================================================
	}

	displayMat(gpuCResult,rA,cB);

	cout << "CPU and GPU diff: " << diffMat(C,gpuCResult,rA,cB) << endl;
	cout << "Value after kernel execution: C[0]: " << gpuCResult[0] << endl;
	
	//wait();
	return 0;
}

TYPE diffMat(TYPE* M1, TYPE* M2, int rM, int cM)
{
	TYPE diff = 0;
	for (int i=0; i<rM; i++)
		for (int j=0; j<cM; j++)
			diff = diff + abs(M2[j+i*cM]-M1[j+i*cM]);
	return diff;
}

void NullMat(TYPE* M, int rM, int cM)
{
	for (int i=0; i<rM; i++)
		for (int j=0; j<cM; j++)
			M[j+i*cM] = 0;
}

void initialiseMat(TYPE* M, int rM, int cM)
{
	for (int i=0; i<rM; i++)
		for (int j=0; j<cM; j++)
			M[j+i*cM] = (TYPE)i+(TYPE)j*((j%3)-1);
}

void displayMat(TYPE* M, int rM, int cM)
{
	// Don't display large matrices
	if (rM > 5 || cM > 5)
		return;
	for (int i=0; i<rM; i++)
	{
		for (int j=0; j<cM; j++)
			cout << M[j+i*cM] << " ";
		cout << endl;
	}
}

void matMult(TYPE* C, TYPE* A, TYPE* B, int rA, int cA, int rB, int cB)
{
	for (int i=0; i<rA; i++)
	{
		for (int j=0; j<cB; j++)
		{
			TYPE Sum = 0;
			for (int k=0; k<cA; k++)
				Sum = Sum+A[k+i*cA]*B[j+k*cB];
			C[j+i*cB] = Sum;
		}
	}
}

void matMultOMP(TYPE* C, TYPE* A, TYPE* B, int rA, int cA, int rB, int cB)
{
#pragma omp parallel //num_threads(16)
	{
#pragma omp for
		for (int i=0; i<rA; i++)
		{
			for (int j=0; j<cB; j++)
			{
				TYPE Sum = 0;
				for (int k=0; k<cA; k++)
					Sum = Sum+A[k+i*cA]*B[j+k*cB];
				C[j+i*cB] = Sum;
			}
		}
	}
}

#ifdef __linux__
void matMultOACC(TYPE* C, TYPE* A, TYPE* B, int rA, int cA, int rB, int cB)
{
#pragma acc data copy(A,B,C)
	{
#pragma acc kernels
		for (int i=0; i<rA; i++)
		{
			for (int j=0; j<cB; j++)
			{
				TYPE Sum = 0;
				for (int k=0; k<cA; k++)
					Sum = Sum+A[k+i*cA]*B[j+k*cB];
				C[j+i*cB] = Sum;
			}
		}
	}
}
#endif
