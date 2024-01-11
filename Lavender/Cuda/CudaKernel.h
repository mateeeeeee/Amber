#pragma once
#include <type_traits>
#include "driver_types.h"

namespace lavender
{
	template<typename KernelType, typename... Args, typename = std::enable_if_t<std::is_invocable_v<KernelType, Args...>, void>>
	void CudaLaunchKernel(KernelType&& kernel, dim3 grid, dim3 block, Args&&... args)
	{
		kernel<<<grid, block>>>(std::forward<Args>(args)...);
	}

	template<typename KernelType, typename... Args, typename = std::enable_if_t<std::is_invocable_v<KernelType, Args...>, void>>
	void CudaLaunchKernel(KernelType&& kernel, dim3 grid, dim3 block, uint64 shared_memory_size, Args&&... args)
	{
		kernel<<<grid, block, shared_memory_size>>>(std::forward<Args>(args)...);
	}

	template<typename KernelType, typename... Args, typename = std::enable_if_t<std::is_invocable_v<KernelType, Args...>, void>>
	cudaError_t CudaLaunchKernel_Debug(KernelType&& kernel, dim3 grid, dim3 block, Args&&... args)
	{
		void* kernel_args[] = { (void*)(&args)... };
		return cudaLaunchKernel((void*)kernel, grid, block, kernel_args, 0, 0);
	}
}