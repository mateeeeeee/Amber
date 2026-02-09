#pragma once
#include "Intersection.h"

namespace amber
{
	template<typename NodeT>
	struct SpatialTraits;

	template<>
	struct SpatialTraits<Triangle>
	{
		static void GrowBounds(AABB& box, Triangle const& node)
		{
			box.Grow(node.v0);
			box.Grow(node.v1);
			box.Grow(node.v2);
		}

		static Float GetCentroid(Triangle const& node, Int axis)
		{
			if (axis == 0) return node.centroid.x;
			if (axis == 1) return node.centroid.y;
			return node.centroid.z;
		}
	};
}
