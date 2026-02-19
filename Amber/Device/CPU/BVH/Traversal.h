#pragma once
#include <algorithm>
#include "BVH.h"
#include "Utilities/Stack.h"

// todo refactor into a single templated Traverse<N> + leaf callback

namespace amber
{
	inline Bool Intersect(BVH2 const& bvh, std::vector<Triangle> const& triangles, Ray& ray, HitInfo& hit)
	{
		if (bvh.nodes.empty()) 
		{
			return false;
		}

		BVH2Node const* node = &bvh.nodes[0];
		SmallStack<BVH2Node const*, 64> stack;
		Bool found = false;
		while (true)
		{
			if (node->IsLeaf())
			{
				for (Uint32 i = 0; i < node->prim_count; i++)
				{
					Uint32 tri_idx = bvh.prim_indices[node->first_prim + i];
					HitInfo temp_hit;
					if (IntersectTriangle(ray, triangles[tri_idx], temp_hit) && temp_hit.t < hit.t)
					{
						hit         = temp_hit;
						hit.tri_idx = tri_idx;
						ray.t       = hit.t;
						found       = true;
						if (Bool(ray.flags & RayFlags::AcceptFirstHit)) 
						{
							return true;
						}
					}
				}
				if (stack.IsEmpty()) 
				{
					break;
				}
				node = stack.Pop();
				continue;
			}

			BVH2Node const* child1 = &bvh.nodes[node->children[0]];
			BVH2Node const* child2 = &bvh.nodes[node->children[1]];
			Float dist1 = IntersectAABB(ray, child1->aabb_min, child1->aabb_max);
			Float dist2 = IntersectAABB(ray, child2->aabb_min, child2->aabb_max);

			if (dist1 > dist2) { std::swap(dist1, dist2); std::swap(child1, child2); }

			if (dist1 == BVH_INFINITY)
			{
				if (stack.IsEmpty()) 
				{
					break;
				}
				node = stack.Pop();
			}
			else
			{
				node = child1;
				if (dist2 != BVH_INFINITY) 
				{
					stack.Push(child2);
				}
			}
		}
		return found;
	}

	template<Uint32 N> requires (N == 4 || N == 8)
	Bool Intersect(BVH<N> const& bvh, std::vector<Triangle> const& triangles, Ray& ray, HitInfo& hit)
	{
		if (bvh.nodes.empty()) 
		{
			return false;
		}

		BVHNode<N> const* node = &bvh.nodes[0];
		SmallStack<BVHNode<N> const*, 64 * N> stack;
		Bool found = false;
		while (true)
		{
			if (node->IsLeaf())
			{
				for (Uint32 i = 0; i < node->prim_count; i++)
				{
					Uint32 tri_idx = bvh.prim_indices[node->first_prim + i];
					HitInfo temp_hit;
					if (IntersectTriangle(ray, triangles[tri_idx], temp_hit) && temp_hit.t < hit.t)
					{
						hit         = temp_hit;
						hit.tri_idx = tri_idx;
						ray.t       = hit.t;
						found       = true;
						if (Bool(ray.flags & RayFlags::AcceptFirstHit)) 
						{
							return true;
						}
					}
				}
				if (stack.IsEmpty()) 
				{
					break;
				}
				node = stack.Pop();
				continue;
			}

			struct ChildHit { BVHNode<N> const* node; Float dist; };
			ChildHit hits[N];
			Uint32 hit_count = 0;
			for (Uint32 i = 0; i < node->child_count; i++)
			{
				BVHNode<N> const* child = &bvh.nodes[node->children[i]];
				Float dist = IntersectAABB(ray, child->aabb_min, child->aabb_max);
				if (dist != BVH_INFINITY) hits[hit_count++] = { child, dist };
			}

			if (hit_count == 0)
			{
				if (stack.IsEmpty()) 
				{
					break;
				}
				node = stack.Pop();
				continue;
			}

			std::sort(hits, hits + hit_count, [](ChildHit const& a, ChildHit const& b) { return a.dist > b.dist; });
			node = hits[hit_count - 1].node;
			for (Uint32 i = 0; i < hit_count - 1; i++) stack.Push(hits[i].node);
		}
		return found;
	}
}
