#include "cuda_runtime.h"
#include "CudaUtil.h"
#include "Core/Logger.h"

namespace lavender
{
	void CudaCheck(cudaError_t code)
	{
		if (code != cudaError::cudaSuccess)
		{
			LAV_ERROR("{}", cudaGetErrorString(code));
			std::exit(EXIT_FAILURE);
		} 
	}
	void CudaCheckKernel()
	{
		cudaError_t code = cudaDeviceSynchronize();
		if (code != cudaError::cudaSuccess)
		{
			LAV_ERROR("Kernel launch failed: {}", cudaGetErrorString(code));
			std::exit(EXIT_FAILURE);
		}
	}
}

