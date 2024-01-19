#pragma once
#include <type_traits>
#include "driver_types.h"
#include "CudaCore.h"
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

		LAV_HOST_DEVICE uint64 GetAllocSize() const { return alloc_size; }

		operator void const* () const
		{
			return dev_alloc;
		}
		operator void* ()
		{
			return dev_alloc;
		}

		template<typename U>
		LAV_HOST_DEVICE U* As()
		{
			return reinterpret_cast<U*>(dev_alloc);
		}
		template<typename U>
		LAV_HOST_DEVICE U const* As() const
		{
			return reinterpret_cast<U*>(dev_alloc);
		}

		void Realloc(uint64 _alloc_size);

	protected:
		void* dev_alloc = nullptr;
		uint64 alloc_size = 0;
	};

	template<typename T>
	class TypedCudaAlloc : public CudaAlloc
	{
	public:
		explicit TypedCudaAlloc(uint64 count) : CudaAlloc(count * sizeof(T)) {}
		LAV_HOST_DEVICE uint64 GetCount() const { return GetAllocSize() / sizeof(T); }

		template<typename U = T>
		LAV_HOST_DEVICE U* As()
		{
			return reinterpret_cast<U*>(dev_alloc);
		}
		template<typename U = T>
		LAV_HOST_DEVICE U const* As() const
		{
			return reinterpret_cast<U*>(dev_alloc);
		}

		operator T* () 
		{
			return reinterpret_cast<T*>(dev_alloc);
		}
		operator T const* ()
		{
			return reinterpret_cast<T const*>(dev_alloc);
		}

		void Realloc(uint64 _alloc_size)
		{
			CudaAlloc::Realloc(_alloc_size * sizeof(T));
		}
	};
}