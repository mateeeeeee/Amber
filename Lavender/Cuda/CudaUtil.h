#include "driver_types.h"

namespace lavender
{
#ifdef __CUDACC__
	#define LAV_CUDA_CALLABLE __host__ __device__
	#define LAV_CUDA_KERNEL __global__
#else
	#define LAV_CUDA_CALLABLE
	#define LAV_CUDA_KERNEL 
#endif

	#define LAV_CUDA_ASSERT(code) assert(code == cudaError::cudaSuccess)


	void CudaCheck(cudaError_t code);
	void CudaCheckKernel();
}