#pragma once
#include "driver_types.h"

namespace lavender
{
	class CudaStream;
	class CudaEvent
	{
	public:
		CudaEvent();
		explicit CudaEvent(uint32 flags);
		LAV_NONCOPYABLE_NONMOVABLE(CudaEvent)
		~CudaEvent();

		void RecordOnStream(CudaStream& stream);
		void Synchronize();
		bool IsDone() const;

		operator cudaEvent_t() const;

	private:
		cudaEvent_t event;
	};
}