#include "cuda_runtime.h"
#include "CudaEvent.h"
#include "CudaStream.h"
#include "CudaUtil.h"

namespace lavender
{

	CudaEvent::CudaEvent(uint32 flags)
	{
		CudaCheck(cudaEventCreateWithFlags(&event, flags));
	}
	CudaEvent::CudaEvent()
	{
		CudaCheck(cudaEventCreate(&event));
	}
	CudaEvent::~CudaEvent()
	{
		CudaCheck(cudaEventDestroy(event));
	}

	void CudaEvent::RecordOnStream()
	{
		CudaCheck(cudaEventRecord(event));
	}

	void CudaEvent::RecordOnStream(CudaStream& stream)
	{
		CudaCheck(cudaEventRecord(event, stream));
	}

	void CudaEvent::Synchronize()
	{
		CudaCheck(cudaEventSynchronize(event));
	}
	bool CudaEvent::IsDone() const
	{
		return cudaEventQuery(event) != cudaErrorNotReady;
	}

	CudaEvent::operator cudaEvent_t() const
	{
		return event;
	}

}

