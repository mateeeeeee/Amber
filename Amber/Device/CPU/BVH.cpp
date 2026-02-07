#include "BVH.h"
#include <algorithm>

namespace amber
{
	static void IntersectRecursiveImpl(BVH const& bvh, Ray const& ray, Uint32 node_idx, HitInfo& hit, Bool& found)
	{
		BVHNode const& node = bvh.nodes[node_idx];
		if (IntersectAABB(ray, node.aabb_min, node.aabb_max) == BVH_INFINITY)
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
			IntersectRecursiveImpl(bvh, ray, node.left_first, hit, found);
			IntersectRecursiveImpl(bvh, ray, node.left_first + 1, hit, found);
		}
	}

	Bool IntersectRecursive(BVH const& bvh, Ray const& ray, HitInfo& hit)
	{
		if (bvh.nodes.empty())
		{
			return false;
		}
		Bool found = false;
		IntersectRecursiveImpl(bvh, ray, 0, hit, found);
		return found;
	}

	Bool Intersect(BVH const& bvh, Ray const& ray, HitInfo& hit)
	{
		if (bvh.nodes.empty())
		{
			return false;
		}

		BVHNode const* node = &bvh.nodes[0];
		BVHNode const* stack[64];
		Uint32 stack_ptr = 0;
		Bool found = false;

		Ray local_ray = ray;
		while (true)
		{
			if (node->IsLeaf())
			{
				for (Uint32 i = 0; i < node->tri_count; i++)
				{
					Uint32 tri_idx = bvh.tri_indices[node->left_first + i];
					Triangle const& tri = (*bvh.triangles)[tri_idx];
					HitInfo temp_hit;
					if (IntersectTriangle(local_ray, tri, temp_hit) && temp_hit.t < hit.t)
					{
						hit = temp_hit;
						hit.tri_idx = tri_idx;
						local_ray.t = hit.t;
						found = true;
					}
				}
				if (stack_ptr == 0) break;
				node = stack[--stack_ptr];
				continue;
			}

			BVHNode const* child1 = &bvh.nodes[node->left_first];
			BVHNode const* child2 = &bvh.nodes[node->left_first + 1];
			Float dist1 = IntersectAABB(local_ray, child1->aabb_min, child1->aabb_max);
			Float dist2 = IntersectAABB(local_ray, child2->aabb_min, child2->aabb_max);

			if (dist1 > dist2)
			{
				std::swap(dist1, dist2);
				std::swap(child1, child2);
			}

			if (dist1 == BVH_INFINITY)
			{
				if (stack_ptr == 0) break;
				node = stack[--stack_ptr];
			}
			else
			{
				node = child1;
				if (dist2 != BVH_INFINITY) stack[stack_ptr++] = child2;
			}
		}

		return found;
	}
}
