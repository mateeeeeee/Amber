#include "cuda_runtime.h"
#include "CudaUtils.h"
#include "Core/Logger.h"


namespace amber::optix
{
	void CudaCheck(cudaError_t code)
	{
		if (code != cudaError::cudaSuccess)
		{
			LAV_ERROR("%s", cudaGetErrorString(code));
			std::exit(EXIT_FAILURE);
		}
	}
	void CudaSyncCheck()
	{
		cudaError_t code = cudaDeviceSynchronize();
		if (code != cudaError::cudaSuccess)
		{
			LAV_ERROR("Kernel launch failed: %s", cudaGetErrorString(code));
			std::exit(EXIT_FAILURE);
		}
	}

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

	CudaStream::CudaStream(uint32 flags)
	{
		CudaCheck(cudaStreamCreateWithFlags(&stream, flags));
	}
	CudaStream::CudaStream() : stream{}
	{
		CudaCheck(cudaStreamCreate(&stream));
	}
	CudaStream::~CudaStream()
	{
		CudaCheck(cudaStreamDestroy(stream));
	}

	void CudaStream::Synchronize() const
	{
		CudaCheck(cudaStreamSynchronize(stream));
	}
	void CudaStream::WaitForEvent(cudaEvent_t event)
	{
		CudaCheck(cudaStreamWaitEvent(stream, event));
	}
	bool CudaStream::IsDone() const
	{
		return cudaStreamQuery(stream) != cudaErrorNotReady;
	}

	CudaStream::operator cudaStream_t() const
	{
		return stream;
	}
}

