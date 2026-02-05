#include "CpuBVH.h"

namespace amber
{
	Bool CpuBVH::Intersect(Ray const& ray, HitInfo& hit) const
	{
		if (data.nodes.empty())
		{
			return false;
		}
		Bool found = false;
		IntersectRecursive(ray, 0, hit, found);
		return found;
	}

	void CpuBVH::IntersectRecursive(Ray const& ray, Uint32 node_idx, HitInfo& hit, Bool& found) const
	{
		BVHNode const& node = data.nodes[node_idx];
		if (!IntersectAABB(ray, node.aabb_min, node.aabb_max, hit.t))
		{
			return;
		}

		if (node.IsLeaf())
		{
			for (Uint32 i = 0; i < node.tri_count; i++)
			{
				Uint32 tri_idx = data.tri_indices[node.left_first + i];
				Triangle const& tri = (*triangles)[tri_idx];
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
			IntersectRecursive(ray, node.left_first, hit, found);
			IntersectRecursive(ray, node.left_first + 1, hit, found);
		}
	}
}
