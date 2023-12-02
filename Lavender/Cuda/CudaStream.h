#pragma once
#include "driver_types.h"

namespace lavender
{
	class CudaStream
	{
	public:
		CudaStream();
		explicit CudaStream(uint32 flags);
		LAVENDER_NONCOPYABLE_NONMOVABLE(CudaStream)
		~CudaStream();

		void Synchronize() const;
		void WaitForEvent(cudaEvent_t event);
		bool IsDone() const;

		operator cudaStream_t() const;

	private:
		cudaStream_t stream;
	};
}