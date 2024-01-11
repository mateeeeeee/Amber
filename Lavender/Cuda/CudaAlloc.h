#pragma once
#include <type_traits>
#include "driver_types.h"
#include "Core/Defines.h"
#include "Core/CoreTypes.h"

namespace lavender
{
	class CudaAlloc
	{
	public:
		explicit CudaAlloc(uint64 alloc_in_bytes);
		LAV_NONCOPYABLE_NONMOVABLE(CudaAlloc)
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

	protected:
		void* dev_alloc = nullptr;
		uint64 const alloc_size = 0;
	};

	template<typename T>
	class TypedCudaAlloc : public CudaAlloc
	{
	public:
		explicit TypedCudaAlloc(uint64 count) : CudaAlloc(count * sizeof(T)) {}
		uint64 GetCount() const { return GetAllocSize() / sizeof(T); }

		template<typename U = T>
		U* As()
		{
			return reinterpret_cast<U*>(dev_alloc);
		}
		template<typename U = T>
		U const* As() const
		{
			return reinterpret_cast<U*>(dev_alloc);
		}
	};
}