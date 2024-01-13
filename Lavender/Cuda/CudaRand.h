#pragma once
#include <curand_kernel.h>
#include "CudaCore.h"
#include "CudaAlloc.h"

namespace lavender
{
	class CudaRand
	{
	public:
		explicit CudaRand(uint64 count);
		LAV_NONCOPYABLE_NONMOVABLE(CudaRand)
		~CudaRand();

		LAV_DEVICE float Generate(uint64 i);
		LAV_HOST   void Realloc(uint64 count);

		LAV_HOST_DEVICE curandState* As()
		{
			return rand_state_alloc.As();
		}
		LAV_HOST_DEVICE curandState const* As() const
		{
			return rand_state_alloc.As();
		}

		LAV_HOST operator curandState* ()
		{
			return rand_state_alloc;
		}
		LAV_HOST operator curandState const* ()
		{
			return rand_state_alloc;
		}

	private:
		TypedCudaAlloc<curandState> rand_state_alloc;
	};
}