#pragma once
#include <memory>
#include <vector>
#include <string>

@protocol MTLDevice;
@protocol MTLCommandQueue;
@protocol MTLBuffer;
@protocol MTLTexture;
@protocol MTLComputePipelineState;
@protocol MTLAccelerationStructure;
@protocol MTLAccelerationStructureCommandEncoder;

namespace amber::metal
{
	class Device
	{
	public:
		Device();
		~Device();

		id<MTLDevice> GetDevice() const { return device; }
		id<MTLCommandQueue> GetCommandQueue() const { return command_queue; }

	private:
		id<MTLDevice> device;
		id<MTLCommandQueue> command_queue;
	};

	class Buffer
	{
	public:
		Buffer(id<MTLDevice> device, Uint64 size, Uint32 options = 0);
		~Buffer();

		id<MTLBuffer> GetBuffer() const { return buffer; }
		Uint64 GetSize() const { return size; }
		void* GetContents() const;

		void Update(void const* data, Uint64 data_size);

		template<typename T>
		void Update(T const& data)
		{
			Update(&data, sizeof(T));
		}

		template<typename T>
		T* As()
		{
			return reinterpret_cast<T*>(GetContents());
		}

		template<typename T>
		T const* As() const
		{
			return reinterpret_cast<T const*>(GetContents());
		}

	private:
		id<MTLBuffer> buffer;
		Uint64 size;
	};

	template<typename T>
	class TBuffer : public Buffer
	{
	public:
		explicit TBuffer(id<MTLDevice> device, Uint64 count = 1, Uint32 options = 0)
			: Buffer(device, count * sizeof(T), options) {}

		Uint64 GetCount() const { return GetSize() / sizeof(T); }

		T* operator->() { return As<T>(); }
		T const* operator->() const { return As<T>(); }
	};

	class Texture2D
	{
	public:
		Texture2D(id<MTLDevice> device, Uint32 width, Uint32 height, Uint32 pixel_format, Bool read_only = false);
		~Texture2D();

		id<MTLTexture> GetTexture() const { return texture; }
		Uint32 GetWidth() const { return width; }
		Uint32 GetHeight() const { return height; }

		void Update(void const* data, Uint32 bytes_per_row);

	private:
		id<MTLTexture> texture;
		Uint32 width;
		Uint32 height;
	};

	class ComputePipeline
	{
	public:
		ComputePipeline(id<MTLDevice> device, Char const* shader_source, Char const* function_name);
		static std::unique_ptr<ComputePipeline> CreateFromFile(id<MTLDevice> device, Char const* file_path, Char const* function_name);
		~ComputePipeline();

		id<MTLComputePipelineState> GetPipelineState() const { return pipeline_state; }

	private:
		id<MTLComputePipelineState> pipeline_state;
	};

	class AccelerationStructure
	{
	public:
		explicit AccelerationStructure(id<MTLDevice> device);
		~AccelerationStructure();

		void BuildPrimitiveAccelerationStructure(
			id<MTLAccelerationStructureCommandEncoder> encoder,
			id<MTLBuffer> vertex_buffer,
			Uint32 vertex_offset,
			Uint32 vertex_count,
			id<MTLBuffer> index_buffer,
			Uint32 index_offset,
			Uint32 triangle_count);

		void BuildInstanceAccelerationStructure(
			id<MTLAccelerationStructureCommandEncoder> encoder,
			void const* instances_data,
			Uint32 instance_count,
			id<MTLAccelerationStructure> const* blas_array,
			Uint32 blas_count);

		id<MTLAccelerationStructure> GetAccelerationStructure() const { return acceleration_structure; }

	private:
		id<MTLDevice> device;
		id<MTLAccelerationStructure> acceleration_structure;
		id<MTLBuffer> scratch_buffer;
	};
}
