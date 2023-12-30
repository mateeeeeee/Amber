#include "driver_types.h"

namespace lavender
{
#ifdef __CUDACC__
	#define LAVENDER_CUDA_CALLABLE __host__ __device__
#else
	#define LAVENDER_CUDA_CALLABLE
#endif

	#define LAVENDER_CUDA_ASSERT(code) assert(code == cudaError::cudaSuccess)


	void CudaCheck(cudaError_t code);
	void CudaCheckKernel();
}