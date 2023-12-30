#include "driver_types.h"

namespace lavender
{
#ifdef __CUDACC__
	#define CUDA_CALLABLE __host__ __device__
	#define CUDA_KERNEL __global__
#else
	#define CUDA_CALLABLE
	#define CUDA_KERNEL 
#endif

	#define CUDA_ASSERT(code) assert(code == cudaError::cudaSuccess)


	void CudaCheck(cudaError_t code);
	void CudaCheckKernel();
}