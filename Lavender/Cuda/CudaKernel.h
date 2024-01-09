#pragma once
#include <type_traits>
#include "driver_types.h"

namespace lavender
{
	struct CudaKernelLauncher
	{
		template<typename KernelType, typename... Args, typename = std::enable_if_t<std::is_invocable_v<KernelType, Args...>, void>>
		void operator()(KernelType&& kernel, dim3 grid, dim3 block, Args&&... args) const
		{
			kernel<<<grid, block>>>(std::forward<Args>(args)...);
		}

		template<typename KernelType, typename... Args, typename = std::enable_if_t<std::is_invocable_v<KernelType, Args...>, void>>
		void operator()(KernelType&& kernel, dim3 grid, dim3 block, uint64 shared_memory_size, Args&&... args) const
		{
			kernel<<<grid, block, shared_memory_size>>>(std::forward<Args>(args)...);
		}
	};

	#define CUDA_LAUNCH(kernel, grid, block, ...) lavender::CudaKernelLauncher{}(kernel, grid, block, __VA_ARGS__)
	#define CUDA_LAUNCH_SHARED(kernel, grid, block, shared, ...) lavender::CudaKernelLauncher{}(kernel, grid, block, shared, __VA_ARGS__)
}