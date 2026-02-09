#pragma once
#include <vector>
#include "BVH/BVH.h"

namespace amber
{
	struct BLAS
	{
		BVH bvh;
		std::vector<Triangle> triangles;
	};

	Bool Intersect(BLAS const& blas, Ray& ray, HitInfo& hit);
	Bool IntersectRecursive(BLAS const& blas, Ray const& ray, HitInfo& hit);
	void Refit(BLAS& blas);
}
