#pragma once
#include "BLAS.h"

namespace amber
{
	struct TLASNode
	{
		Vector3 aabb_min;
		Uint32 left_first;
		Vector3 aabb_max;
		Uint32 is_leaf;
	};

	struct TLAS
	{
		std::vector<TLASNode> nodes;
		BLAS* blas_list = nullptr;
		Uint32 blas_count = 0;
		Uint32 nodes_used = 0;
	};

	void Build(TLAS& tlas, BLAS* blas_list, Uint32 blas_count);
	Bool Intersect(TLAS const& tlas, Ray& ray, HitInfo& hit);
}
