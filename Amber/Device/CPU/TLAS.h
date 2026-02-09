#pragma once
#include <vector>
#include "BLAS.h"
#include "BVH/BVH.h"
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
		BVH                      bvh;
		std::vector<BLASInstance> instances;
	};

	Bool Intersect(BLASInstance const& instance, Uint32 instance_idx, Ray& ray, HitInfo& hit);
	Bool Intersect(TLAS const& tlas, Ray& ray, HitInfo& hit);
}
