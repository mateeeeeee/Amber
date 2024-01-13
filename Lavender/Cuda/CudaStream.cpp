#include "cuda_runtime.h"
#include "CudaStream.h"
#include "CudaCore.h"

namespace lavender
{

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

