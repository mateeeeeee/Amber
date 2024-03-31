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

	Texture2D::Texture2D(uint32 w, uint32 h, cudaChannelFormatDesc format, bool srgb) : width(w), height(h), format(format)
	{
		CudaCheck(cudaMallocArray(&data, &format, width, height));

		cudaResourceDesc res_desc{};
		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array = data;

		cudaTextureDesc tex_desc = {};
		tex_desc.addressMode[0] = cudaAddressModeWrap;
		tex_desc.addressMode[1] = cudaAddressModeWrap;
		tex_desc.filterMode = cudaFilterModeLinear;
		tex_desc.readMode = cudaReadModeNormalizedFloat;
		tex_desc.sRGB = srgb ? 1 : 0;
		tex_desc.normalizedCoords = 1;
		tex_desc.maxAnisotropy = 1;
		tex_desc.maxMipmapLevelClamp = 1;
		tex_desc.minMipmapLevelClamp = 1;
		tex_desc.mipmapFilterMode = cudaFilterModePoint;
		CudaCheck(cudaCreateTextureObject(&texture_handle, &res_desc, &tex_desc, nullptr));
	}


	Texture2D::Texture2D(Texture2D&& texture) noexcept
		: width(texture.width), height(texture.height), format(texture.format), 
		  data(texture.data), texture_handle(texture.texture_handle)
	{
		texture.width = 0;
		texture.height = 0;
		texture.data = 0;
		texture.texture_handle = 0;
	}

	Texture2D& Texture2D::operator=(Texture2D&& texture) noexcept
	{
		if (data) 
		{
			cudaFreeArray(data);
			cudaDestroyTextureObject(texture_handle);
		}
		width = texture.width;
		height = texture.height;
		format = texture.format;
		data = texture.data;
		texture_handle = texture.texture_handle;

		texture.width = 0;
		texture.height = 0;
		texture.data = 0;
		texture.texture_handle = 0;
		return *this;
	}

	Texture2D::~Texture2D()
	{
		if (data) 
		{
			CudaCheck(cudaFreeArray(data));
			CudaCheck(cudaDestroyTextureObject(texture_handle));
		}
	}

	void Texture2D::Update(void const* new_data)
	{
		uint64 pixel_size = (format.x + format.y + format.z + format.w) / 8;
		uint64 pitch = pixel_size * width;
		CudaCheck(cudaMemcpy2DToArray(data, 0, 0, new_data, pitch, pitch, height, cudaMemcpyHostToDevice));
	}

}

