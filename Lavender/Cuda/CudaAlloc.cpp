#include "cuda_runtime.h"
#include "CudaAlloc.h"
#include "CudaUtil.h"

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

}
