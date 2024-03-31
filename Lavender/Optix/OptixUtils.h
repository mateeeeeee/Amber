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
		

	private:
	};
}