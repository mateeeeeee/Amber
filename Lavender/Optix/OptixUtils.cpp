#include <cuda_runtime.h>
#include <optix_stubs.h>
#include "OptixUtils.h"
#include "CudaUtils.h"
#include "Core/Logger.h"

namespace lavender::optix
{

	void OptixCheck(OptixResult code)
	{
		if (code != OPTIX_SUCCESS)
		{
			LAV_ERROR("%s", optixGetErrorString(code));
			std::exit(EXIT_FAILURE);
		}
	}

	Buffer::Buffer(uint64 size) 
	{
		CudaCheck(cudaMalloc(&dev_ptr, size));
	}

	Buffer::Buffer(Buffer&& buffer) noexcept : size(buffer.size), dev_ptr(buffer.dev_ptr)
	{
		buffer.size = 0;
		buffer.dev_ptr = nullptr;
	}

	Buffer& Buffer::operator=(Buffer&& buffer) noexcept
	{
		CudaCheck(cudaFree(dev_ptr));
		size = buffer.size;
		dev_ptr = buffer.dev_ptr;

		buffer.size = 0;
		buffer.dev_ptr = nullptr;
		return *this;
	}

	Buffer::~Buffer()
	{
		CudaCheck(cudaFree(dev_ptr));
	}

	void Buffer::Update(void const* data, uint64 data_size)
	{
		CudaCheck(cudaMemcpy(dev_ptr, data, data_size, cudaMemcpyHostToDevice));
	}

}

