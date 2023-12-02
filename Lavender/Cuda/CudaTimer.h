#pragma once
#include "CudaEvent.h"

namespace lavender
{
	class CudaTimer
	{
	public:
		void Start()
		{
			start.RecordOnStream();
			kernel << <grid, block, shared_memory_size >> > (std::forward<Args>(args)...);
			stop.RecordOnStream();
			stop.Synchronize();
			float32 ms;
			cudaEventElapsedTime(&ms, start, stop);
		}

	private:
		CudaEvent start, stop;
	};
}