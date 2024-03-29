#include "cuda_runtime.h"
#include "CudaAlloc.h"
#include "CudaCore.h"

namespace lavender
{

	CudaAlloc::CudaAlloc(uint64 alloc_in_bytes) : alloc_size(alloc_in_bytes)
	{
		CudaCheck(cudaMalloc(&dev_alloc, alloc_in_bytes));
	}

	CudaAlloc::~CudaAlloc()
	{
		CudaCheck(cudaFree(dev_alloc));
	}

	void CudaAlloc::Realloc(uint64 _alloc_size)
	{
		if (alloc_size != _alloc_size)
		{
			CudaCheck(cudaFree(dev_alloc));
			CudaCheck(cudaMalloc(&dev_alloc, _alloc_size));
			alloc_size = _alloc_size;
		}
	}

}
