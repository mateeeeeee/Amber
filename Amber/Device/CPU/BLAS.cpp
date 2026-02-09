#include "BLAS.h"

namespace amber
{
	BLAS::BLAS(BLAS const& other)
		: bvh(other.bvh), triangles(other.triangles)
	{
		bvh.triangles = &triangles;
	}

	BLAS::BLAS(BLAS&& other) noexcept
		: bvh(std::move(other.bvh)), triangles(std::move(other.triangles))
	{
		bvh.triangles = &triangles;
	}

	BLAS& BLAS::operator=(BLAS const& other)
	{
		if (this != &other)
		{
			bvh = other.bvh;
			triangles = other.triangles;
			bvh.triangles = &triangles;
		}
		return *this;
	}

	BLAS& BLAS::operator=(BLAS&& other) noexcept
	{
		if (this != &other)
		{
			bvh = std::move(other.bvh);
			triangles = std::move(other.triangles);
			bvh.triangles = &triangles;
		}
		return *this;
	}
}
