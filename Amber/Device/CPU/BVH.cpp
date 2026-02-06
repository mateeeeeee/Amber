#include "BVH.h"

namespace amber
{
	static void IntersectRecursive(BVH const& bvh, Ray const& ray, Uint32 node_idx, HitInfo& hit, Bool& found)
	{
		BVHNode const& node = bvh.nodes[node_idx];
		if (!IntersectAABB(ray, node.aabb_min, node.aabb_max, hit.t))
		{
			return;
		}

		if (node.IsLeaf())
		{
			for (Uint32 i = 0; i < node.tri_count; i++)
			{
				Uint32 tri_idx = bvh.tri_indices[node.left_first + i];
				Triangle const& tri = (*bvh.triangles)[tri_idx];
				HitInfo temp_hit;
				if (IntersectTriangle(ray, tri, temp_hit) && temp_hit.t < hit.t)
				{
					hit = temp_hit;
					hit.tri_idx = tri_idx;
					found = true;
				}
			}
		}
		else
		{
			IntersectRecursive(bvh, ray, node.left_first, hit, found);
			IntersectRecursive(bvh, ray, node.left_first + 1, hit, found);
		}
	}

	Bool Intersect(BVH const& bvh, Ray const& ray, HitInfo& hit)
	{
		if (bvh.nodes.empty())
		{
			return false;
		}
		Bool found = false;
		IntersectRecursive(bvh, ray, 0, hit, found);
		return found;
	}
}
