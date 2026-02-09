#pragma once
#include <vector>
#include "BVH/BVH.h"

namespace amber
{
	struct BLAS
	{
		BVH bvh;
		std::vector<Triangle> triangles;

		BLAS() = default;
		BLAS(BLAS const& other);
		BLAS(BLAS&& other) noexcept;
		BLAS& operator=(BLAS const& other);
		BLAS& operator=(BLAS&& other) noexcept;
	};
}
