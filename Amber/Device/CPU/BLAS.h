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

		BLAS(BLAS const& other)
			: bvh(other.bvh), triangles(other.triangles),
			  transform(other.transform), inv_transform(other.inv_transform),
			  world_bounds(other.world_bounds)
		{
			bvh.triangles = &triangles;
		}

		BLAS(BLAS&& other) noexcept
			: bvh(std::move(other.bvh)), triangles(std::move(other.triangles)),
			  transform(other.transform), inv_transform(other.inv_transform),
			  world_bounds(other.world_bounds)
		{
			bvh.triangles = &triangles;
		}

		BLAS& operator=(BLAS const& other)
		{
			if (this != &other)
			{
				bvh = other.bvh;
				triangles = other.triangles;
				transform = other.transform;
				inv_transform = other.inv_transform;
				world_bounds = other.world_bounds;
				bvh.triangles = &triangles;
			}
			return *this;
		}

		BLAS& operator=(BLAS&& other) noexcept
		{
			if (this != &other)
			{
				bvh = std::move(other.bvh);
				triangles = std::move(other.triangles);
				transform = other.transform;
				inv_transform = other.inv_transform;
				world_bounds = other.world_bounds;
				bvh.triangles = &triangles;
			}
			return *this;
		}

		void SetTransform(Matrix const& t);
	};

	void Build(BLAS& blas);
	Bool Intersect(BLAS const& blas, Uint32 blas_idx, Ray& ray, HitInfo& hit);
}
