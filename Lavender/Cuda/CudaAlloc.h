#pragma once
#include <type_traits>
#include "driver_types.h"

namespace lavender
{
	class CudaAlloc
	{
	public:
		explicit CudaAlloc(uint64 alloc_in_bytes);
		LAVENDER_NONCOPYABLE_NONMOVABLE(CudaAlloc)
		~CudaAlloc();

		uint64 GetAllocSize() const { return alloc_size; }

		operator void const* () const
		{
			return dev_alloc;
		}
		operator void* ()
		{
			return dev_alloc;
		}

		template<typename U>
		U* As()
		{
			return reinterpret_cast<U*>(dev_alloc);
		}
		template<typename U>
		U const* As() const
		{
			return reinterpret_cast<U*>(dev_alloc);
		}

	private:
		void* dev_alloc = nullptr;
		uint64 const alloc_size = 0;
	};

	template<typename T>
	class TypedCudaAlloc : public CudaAlloc
	{
	public:
		explicit TypedCudaAlloc(uint64 count) : CudaAlloc(count * sizeof(T)) {}
		uint64 GetCount() const { return GetAllocSize() / sizeof(T); }

		template<typename U>
		U* As()
		{
			static_assert(std::is_same_v<T,U>);
			return reinterpret_cast<U*>(dev_alloc);
		}
		template<typename U>
		U const* As() const
		{
			static_assert(std::is_same_v<T, U>);
			return reinterpret_cast<U*>(dev_alloc);
		}
	};
}