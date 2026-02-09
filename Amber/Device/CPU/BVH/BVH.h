#pragma once
#include <vector>
#include "Device/CPU/BVH/Intersection.h"

namespace amber
{
	struct BVHNode
	{
		Vector3 aabb_min;
		Vector3 aabb_max;
		Uint32 left_first;
		Uint32 tri_count;

		Bool IsLeaf() const { return tri_count > 0; }
	};

	struct BVH
	{
		std::vector<BVHNode> nodes;
		std::vector<Uint32>  prim_indices;
		Uint32               nodes_used = 0;
	};
}
