#pragma once
#include "TopDownBuilder.h"
#include "Device/CPU/BVH/Primitives.h"
#include "Device/CPU/TLAS.h"

namespace amber
{
	struct MedianSplitPolicy
	{
		template<typename NodeT>
		static std::optional<SplitResult> FindSplit(BVH const& bvh, NodeT const* nodes, BVHNode const& node)
		{
			Vector3 extent = node.aabb_max - node.aabb_min;
			Int axis = 0;
			if (extent.y > extent.x) axis = 1;
			if (extent.z > (axis == 0 ? extent.x : extent.y)) axis = 2;

			Float pos;
			if (axis == 0)      pos = node.aabb_min.x + extent.x * 0.5f;
			else if (axis == 1) pos = node.aabb_min.y + extent.y * 0.5f;
			else                pos = node.aabb_min.z + extent.z * 0.5f;

			return SplitResult{ axis, pos };
		}
	};

	using MedianSplitBuilderBLAS = TopDownBuilder<Triangle,     MedianSplitPolicy>;
	using MedianSplitBuilderTLAS = TopDownBuilder<BLASInstance, MedianSplitPolicy>;
}
