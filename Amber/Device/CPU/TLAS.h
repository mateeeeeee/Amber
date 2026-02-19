#pragma once
#include <vector>
#include "BLAS.h"
#include "BVH/BVH.h"
#include "BVH/SpatialTraits.h"
#include "RayFlags.h"

namespace amber
{
	struct BLASInstance
	{
		BLAS const* blas         = nullptr;
		Matrix      inv_transform;
		AABB        world_bounds;
		Uint32      instance_id  = 0;
		Uint8       mask         = 0xFF;
		RayGeometryFlags flags   = RayGeometryFlags::Opaque;
	};

	struct TLAS
	{
		BVH8                      bvh;
		std::vector<BLASInstance> instances;
	};

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

	Bool Intersect(TLAS const& tlas, Ray& ray, HitInfo& hit);
}

