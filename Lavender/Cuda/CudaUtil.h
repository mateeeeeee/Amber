#include "driver_types.h"

namespace lavender
{
	#define LAV_CUDA_CALLABLE __host__ __device__
	#define LAV_CUDA_KERNEL __global__

	#define LAV_CUDA_ASSERT(code) assert(code == cudaError::cudaSuccess)


	void CudaCheck(cudaError_t code);
	void CudaCheckKernel();
}