#pragma once
#include <algorithm>
#include "BVH.h"

namespace amber
{
	template<Uint32 N> requires (N == 2 || N == 4 || N == 8)
	void Refit(BVH<N>& bvh, std::vector<Triangle> const& triangles)
	{
		for (Int i = static_cast<Int>(bvh.nodes_used) - 1; i >= 0; i--)
		{
			BVHNode<N>& node = bvh.nodes[i];
			if (node.IsLeaf())
			{
				node.aabb_min = Vector3(BVH_INFINITY, BVH_INFINITY, BVH_INFINITY);
				node.aabb_max = Vector3(-BVH_INFINITY, -BVH_INFINITY, -BVH_INFINITY);
				for (Uint32 j = 0; j < node.prim_count; j++)
				{
					Triangle const& tri = triangles[bvh.prim_indices[node.first_prim + j]];
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
				node.aabb_min = Vector3(BVH_INFINITY, BVH_INFINITY, BVH_INFINITY);
				node.aabb_max = Vector3(-BVH_INFINITY, -BVH_INFINITY, -BVH_INFINITY);
				for (Uint32 j = 0; j < node.child_count; j++)
				{
					BVHNode<N> const& child = bvh.nodes[node.children[j]];
					node.aabb_min.x = std::min(node.aabb_min.x, child.aabb_min.x);
					node.aabb_min.y = std::min(node.aabb_min.y, child.aabb_min.y);
					node.aabb_min.z = std::min(node.aabb_min.z, child.aabb_min.z);
					node.aabb_max.x = std::max(node.aabb_max.x, child.aabb_max.x);
					node.aabb_max.y = std::max(node.aabb_max.y, child.aabb_max.y);
					node.aabb_max.z = std::max(node.aabb_max.z, child.aabb_max.z);
				}
			}
		}
	}
}
