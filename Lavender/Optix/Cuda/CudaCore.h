#include "driver_types.h"

namespace lavender
{
	#define LAV_KERNEL __global__
	#define LAV_DEVICE __device__
	#define LAV_HOST __host__
	#define LAV_HOST_DEVICE __host__ __device__
	#define LAV_CONSTANT __constant__

	#define LAV_CUDA_ASSERT(code) assert(code == cudaError::cudaSuccess)

	void CudaCheck(cudaError_t code);
	void CudaCheckKernel();
}