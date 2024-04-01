#pragma once
#include <type_traits>
#include "driver_types.h"

#define LAV_KERNEL __global__
#define LAV_DEVICE __device__
#define LAV_HOST __host__
#define LAV_HOST_DEVICE __host__ __device__
#define LAV_CONSTANT __constant__
#define LAV_CUDA_ASSERT(code) LAV_ASSERT(code == cudaError::cudaSuccess)

namespace lavender::optix
{
	void CudaCheck(cudaError_t code);
	void CudaSynchronize();

	class CudaStream;
	class CudaEvent
	{
	public:
		CudaEvent();
		explicit CudaEvent(uint32 flags);
		LAV_NONCOPYABLE_NONMOVABLE(CudaEvent)
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
		LAV_NONCOPYABLE_NONMOVABLE(CudaStream)
		~CudaStream();

		void Synchronize() const;
		void WaitForEvent(cudaEvent_t event);
		bool IsDone() const;

		operator cudaStream_t() const;

	private:
		cudaStream_t stream;
	};
}