#pragma once
#include "BLAS.h"
#include "BVH/PrimTraits.h"

namespace amber
{
	template<>
	struct PrimTraits<BLAS>
	{
		static void GrowBounds(AABB& box, BLAS const& prim)
		{
			box.Grow(prim.world_bounds);
		}

		static Float GetCentroid(BLAS const& prim, Int axis)
		{
			if (axis == 0) return (prim.world_bounds.min.x + prim.world_bounds.max.x) * 0.5f;
			if (axis == 1) return (prim.world_bounds.min.y + prim.world_bounds.max.y) * 0.5f;
			return (prim.world_bounds.min.z + prim.world_bounds.max.z) * 0.5f;
		}
	};
}
