#pragma once
#include "BLAS.h"
#include "BVH/BVH.h"

namespace amber
{
	struct TLAS
	{
		BVH   bvh;
		BLAS* blas_list  = nullptr;
		Uint32 blas_count = 0;
	};

	Bool Intersect(TLAS const& tlas, Ray& ray, HitInfo& hit);
}
