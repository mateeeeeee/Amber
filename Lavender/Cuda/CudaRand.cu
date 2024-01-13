#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CudaRand.h"
#include "CudaKernel.h"

namespace lavender
{
	namespace 
	{
		static constexpr uint64 BLOCK_DIM = 16;
		LAV_KERNEL void CudaRandInit(curandState* rand_state, uint64 size)
		{
			uint64 j = blockIdx.x * blockDim.x + threadIdx.x;
			if (j >= size) return;
			curand_init(1984, j, 0, &rand_state[j]);
		}
	}

	CudaRand::CudaRand(uint64 count) : rand_state_alloc(count)
	{
		dim3 block(BLOCK_DIM, 1, 1);
		dim3 grid((uint32)std::ceil(count / block.x), 1, 1);
		CudaLaunchKernel(CudaRandInit, grid, block, rand_state_alloc, rand_state_alloc.GetCount());
		CudaCheck(cudaDeviceSynchronize());
	}

	CudaRand::~CudaRand() = default;

	LAV_DEVICE float CudaRand::Generate(uint64 i)
	{
		return 0.0f; //curand_uniform(&rand_state_alloc.As()[i]);
	}

	void CudaRand::Realloc(uint64 count)
	{
		if (rand_state_alloc.GetCount() != count)
		{
			rand_state_alloc.Realloc(count);
			dim3 block(BLOCK_DIM, 1, 1);
			dim3 grid((uint32)std::ceil(count / block.x), 1, 1);
			CudaLaunchKernel(CudaRandInit, grid, block, rand_state_alloc, count);
			CudaCheck(cudaDeviceSynchronize());
		}
	}

}

