#pragma once
#include "Core/Types.h"
#include <type_traits>

namespace amber
{
	template<Bool Profile>
	struct KernelLauncher
	{
		template<typename KernelType, typename... Args, typename = std::enable_if_t<std::is_invocable_v<KernelType, Args...>, void>>
		void operator()(KernelType&& kernel, dim3 grid, dim3 block, Args&&... args) const
		{
			kernel<<<grid, block>>>(std::forward<Args>(args)...);
		}

		template<typename KernelType, typename... Args, typename = std::enable_if_t<std::is_invocable_v<KernelType, Args...>, void>>
		void operator()(KernelType&& kernel, dim3 grid, dim3 block, size_t shared_memory_size, Args&&... args) const
		{
			kernel<<<grid, block, shared_memory_size>>>(std::forward<Args>(args)...);
		}
	};

#if USE_KERNEL_PROFILING
	using Launcher = KernelLauncher<true>;
#else 
	using Launcher = KernelLauncher<false>;
#endif

#define LAUNCH_KERNEL(kernel, grid, block, ...) amber::Launcher{}(kernel, grid, block, __VA_ARGS__)
#define LAUNCH_KERNEL_WITH_SHARED(kernel, grid, block, shared, ...) amber::Launcher{}(kernel, grid, block, shared, __VA_ARGS__)
}