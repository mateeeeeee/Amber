#pragma once
#include "TLAS.h"
#include "BVH/SpatialTraits.h"

namespace amber
{
	template<>
	struct SpatialTraits<BLASInstance>
	{
		static void GrowBounds(AABB& box, BLASInstance const& node)
		{
			box.Grow(node.world_bounds);
		}

		static Float GetCentroid(BLASInstance const& node, Int axis)
		{
			if (axis == 0) return (node.world_bounds.min.x + node.world_bounds.max.x) * 0.5f;
			if (axis == 1) return (node.world_bounds.min.y + node.world_bounds.max.y) * 0.5f;
			return (node.world_bounds.min.z + node.world_bounds.max.z) * 0.5f;
		}
	};
}
