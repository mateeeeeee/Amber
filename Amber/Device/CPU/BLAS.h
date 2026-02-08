#pragma once
#include <vector>
#include "BVH/BVH.h"

namespace amber
{
	struct BLAS
	{
		BVH bvh;
		std::vector<Triangle> triangles;
		Matrix transform;
		Matrix inv_transform;
		AABB world_bounds;

		BLAS() = default;
		BLAS(BLAS const& other);
		BLAS(BLAS&& other) noexcept;
		BLAS& operator=(BLAS const& other);
		BLAS& operator=(BLAS&& other) noexcept;

		void SetTransform(Matrix const& t);
	};

	Bool Intersect(BLAS const& blas, Uint32 blas_idx, Ray& ray, HitInfo& hit);
}
