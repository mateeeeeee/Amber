#pragma once
#include <type_traits>
#include <driver_types.h>

#define AMBER_CUDA_ASSERT(code) AMBER_ASSERT(code == cudaError::cudaSuccess)

namespace amber::optix
{
	void CudaCheck(cudaError_t code);
	void CudaSyncCheck();

	class CudaStream;
	class CudaEvent
	{
	public:
		CudaEvent();
		explicit CudaEvent(uint32 flags);
		AMBER_NONCOPYABLE_NONMOVABLE(CudaEvent)
		~CudaEvent();

		void RecordOnStream();
		void RecordOnStream(CudaStream& stream);
		void Synchronize();
		bool IsDone() const;

		operator cudaEvent_t() const;

	private:
		cudaEvent_t event;
	};
	class CudaStream
	{
	public:
		CudaStream();
		explicit CudaStream(uint32 flags);
		AMBER_NONCOPYABLE_NONMOVABLE(CudaStream)
		~CudaStream();

		void Synchronize() const;
		void WaitForEvent(cudaEvent_t event);
		bool IsDone() const;

		operator cudaStream_t() const;

	private:
		cudaStream_t stream;
	};
}