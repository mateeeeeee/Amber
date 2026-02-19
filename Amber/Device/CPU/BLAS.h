#pragma once
#include <vector>
#include "BVH/BVH.h"

namespace amber
{
	struct BLAS
	{
		BVH8 bvh;
		std::vector<Triangle>  triangles;
		std::vector<Uint32>    face_indices;
	};

	Bool Intersect(BLAS const& blas, Ray& ray, HitInfo& hit);
	Bool IntersectRecursive(BLAS const& blas, Ray const& ray, HitInfo& hit);
	void Refit(BLAS& blas);
}
