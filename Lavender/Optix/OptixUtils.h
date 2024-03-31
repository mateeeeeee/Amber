#pragma once
#include <optix.h>
#include <optix_types.h>

namespace lavender::optix
{
	void OptixCheck(OptixResult code);

	class Buffer
	{
	public:
		Buffer() = default;
		Buffer(uint64 size);
		LAV_NONCOPYABLE(Buffer)
		Buffer(Buffer&&) noexcept;
		Buffer& operator=(Buffer&&) noexcept;
		~Buffer();

		uint64 GetSize() const { return size; }

		operator void const* () const
		{
			return dev_ptr;
		}
		operator void* ()
		{
			return dev_ptr;
		}

		template<typename U>
		U* As()
		{
			return reinterpret_cast<U*>(dev_ptr);
		}
		template<typename U>
		U const* As() const
		{
			return reinterpret_cast<U const*>(dev_ptr);
		}

		void Update(void const* data, uint64 data_size);

	protected:
		void* dev_ptr = nullptr;
		uint64 size = 0;
	};

	template<typename T>
	class TypedBuffer : public Buffer
	{
	public:
		explicit TypedBuffer(uint64 count) : Buffer(count * sizeof(T)) {}

		uint64 GetCount() const { return GetSize() / sizeof(T); }

		template<typename U> requires std::is_same_v<T,U>
		U* As()
		{
			return reinterpret_cast<U*>(dev_ptr);
		}
		template<typename U = T>  requires std::is_same_v<T, U>
		U const* As() const
		{
			return reinterpret_cast<U*>(dev_ptr);
		}

		operator T* ()
		{
			return reinterpret_cast<T*>(dev_ptr);
		}
		operator T const* ()
		{
			return reinterpret_cast<T const*>(dev_ptr);
		}
	};

	class Texture2D
	{
	public:
		template<typename Format>
		Texture2D(uint32 w, uint32 h, bool srgb = false) : Texture2D(w,h, cudaCreateChannelDesc<Format>(), srgb)
		{}

		LAV_NONCOPYABLE(Texture2D)
		Texture2D(Texture2D&& t) noexcept;
		Texture2D& operator=(Texture2D&& t) noexcept;
		~Texture2D();

		uint32 GetWidth() const { return width; }
		uint32 GetHeight() const { return height; }
		auto   GetHandle() const { return texture_handle; }

		void Update(void const* data);

	private:
		uint32 width;
		uint32 height;
		cudaChannelFormatDesc format;
		cudaArray_t data = 0;
		cudaTextureObject_t texture_handle = 0;

	private:
		Texture2D(uint32 w, uint32 h, cudaChannelFormatDesc format, bool srgb);
	};


	template <typename T>
	struct ShaderRecord
	{
		alignas(OPTIX_SBT_RECORD_ALIGNMENT) uint8 header[OPTIX_SBT_RECORD_HEADER_SIZE];
		T data;
	};

	class Module
	{
	public:


	private:
	};
}