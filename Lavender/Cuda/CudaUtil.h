#include "driver_types.h"

namespace lavender
{
	#define LAVENDER_CUDA_ASSERT(code) assert(code == cudaError::cudaSuccess)

	void CudaCheck(cudaError_t code);
	void CudaCheckKernel();
}