#pragma once
#ifdef __INTELLISENSE__
	#ifndef __CUDACC__
		#define __CUDACC__
	#else
		static_assert(false);
	#endif
#endif
#include <cuda_runtime.h>
#include <optix.h>
#include "DeviceTypes.cuh"

__forceinline__ __device__ Uint32 PackPointer0(void* ptr)
{
	Uintptr uptr = reinterpret_cast<Uintptr>(ptr);
	return static_cast<Uint32>(uptr >> 32);
}
__forceinline__ __device__ Uint32 PackPointer1(void* ptr)
{
	Uintptr uptr = reinterpret_cast<Uintptr>(ptr);
	return static_cast<Uint32>(uptr);
}

template <typename T>
__device__ __forceinline__ T* GetPayload()
{
	Uint32 p0 = optixGetPayload_0(), p1 = optixGetPayload_1();
	const Uintptr uptr = (Uintptr(p0) << 32) | p1;
	return reinterpret_cast<T*>(uptr);
}


template <typename... Args>
__device__ __forceinline__ void Trace(
	OptixTraversableHandle traversable,
	Float3 ray_origin,
	Float3 ray_direction,
	Float tmin,
	Float tmax, Args&&... payload)
{
	optixTrace(traversable, ray_origin, ray_direction,
		tmin, tmax, 0.0f,
		OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0,
		0,
		0,
		std::forward<Args>(payload)...);
}

__device__ __forceinline__ Bool TraceOcclusion(
	OptixTraversableHandle handle,
	Float3                 ray_origin,
	Float3                 ray_direction,
	Float                  tmin,
	Float                  tmax
)
{
	optixTraverse(
		handle,
		ray_origin,
		ray_direction,
		tmin,
		tmax, 0.0f,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
		1,
		1,
		0
	);
	return optixHitObjectIsHit();
}

template<typename KernelType, typename... Args>
__device__ __host__ __forceinline__ void LaunchKernel(KernelType&& kernel, dim3 grid, dim3 block, Args&&... args)
{
	kernel<<<grid, block>>>(std::forward<Args>(args)...);
}
template<typename KernelType, typename... Args>
__device__ __host__ __forceinline__ void LaunchKernel(KernelType&& kernel, dim3 grid, dim3 block, size_t shared_memory_size, Args&&... args)
{
	kernel<<<grid, block, shared_memory_size>>>(std::forward<Args>(args)...);
}
#define LAUNCH_KERNEL(kernel, grid, block, ...) LaunchKernel(kernel, grid, block, __VA_ARGS__)
#define LAUNCH_KERNEL_WITH_SHARED(kernel, grid, block, shared, ...) LaunchKernel(kernel, grid, block, shared, __VA_ARGS__)
