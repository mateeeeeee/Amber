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

	void Refit(BVH& bvh)
	{
		for (Int i = static_cast<Int>(bvh.nodes_used) - 1; i >= 0; i--)
		{
			if (i == 1) 
			{
				continue;
			}
			
			BVHNode& node = bvh.nodes[i];
			if (node.IsLeaf())
			{
				node.aabb_min = Vector3(BVH_INFINITY, BVH_INFINITY, BVH_INFINITY);
				node.aabb_max = Vector3(-BVH_INFINITY, -BVH_INFINITY, -BVH_INFINITY);
				for (Uint32 j = 0; j < node.tri_count; j++)
				{
					Uint32 tri_idx = bvh.tri_indices[node.left_first + j];
					Triangle const& tri = (*bvh.triangles)[tri_idx];
					node.aabb_min.x = std::min(node.aabb_min.x, std::min(tri.v0.x, std::min(tri.v1.x, tri.v2.x)));
					node.aabb_min.y = std::min(node.aabb_min.y, std::min(tri.v0.y, std::min(tri.v1.y, tri.v2.y)));
					node.aabb_min.z = std::min(node.aabb_min.z, std::min(tri.v0.z, std::min(tri.v1.z, tri.v2.z)));
					node.aabb_max.x = std::max(node.aabb_max.x, std::max(tri.v0.x, std::max(tri.v1.x, tri.v2.x)));
					node.aabb_max.y = std::max(node.aabb_max.y, std::max(tri.v0.y, std::max(tri.v1.y, tri.v2.y)));
					node.aabb_max.z = std::max(node.aabb_max.z, std::max(tri.v0.z, std::max(tri.v1.z, tri.v2.z)));
				}
			}
			else
			{
				BVHNode const& left  = bvh.nodes[node.left_first];
				BVHNode const& right = bvh.nodes[node.left_first + 1];
				node.aabb_min.x = std::min(left.aabb_min.x, right.aabb_min.x);
				node.aabb_min.y = std::min(left.aabb_min.y, right.aabb_min.y);
				node.aabb_min.z = std::min(left.aabb_min.z, right.aabb_min.z);
				node.aabb_max.x = std::max(left.aabb_max.x, right.aabb_max.x);
				node.aabb_max.y = std::max(left.aabb_max.y, right.aabb_max.y);
				node.aabb_max.z = std::max(left.aabb_max.z, right.aabb_max.z);
			}
		}
	}
}
