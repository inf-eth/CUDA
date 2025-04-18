#include <kernels.h>
__global__ void test_print_kernel()
{
	printf("Hello World!\n");
}

void test_print()
{
	test_print_kernel<<<1, 1>>>();
	return;
}