#pragma once
#include "Intersection.h"

namespace amber
{
	template<typename PrimitiveT>
	struct PrimTraits;

	template<>
	struct PrimTraits<Triangle>
	{
		static void GrowBounds(AABB& box, Triangle const& prim)
		{
			box.Grow(prim.v0);
			box.Grow(prim.v1);
			box.Grow(prim.v2);
		}

		static Float GetCentroid(Triangle const& prim, Int axis)
		{
			if (axis == 0) return prim.centroid.x;
			if (axis == 1) return prim.centroid.y;
			return prim.centroid.z;
		}
	};
}
