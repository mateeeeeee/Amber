#include "BLAS.h"
#include "BVH/Traversal.h"
#include <algorithm>

namespace amber
{
	static void IntersectRecursiveImpl(BVH8 const& bvh, std::vector<Triangle> const& triangles, Ray const& ray, Uint32 node_idx, HitInfo& hit, Bool& found)
	{
		BVH8Node const& node = bvh.nodes[node_idx];
		if (IntersectAABB(ray, node.aabb_min, node.aabb_max) == BVH_INFINITY)
		{
			return;
		}

		if (node.IsLeaf())
		{
			for (Uint32 i = 0; i < node.prim_count; i++)
			{
				Uint32 tri_idx = bvh.prim_indices[node.first_prim + i];
				Triangle const& tri = triangles[tri_idx];
				HitInfo temp_hit;
				if (IntersectTriangle(ray, tri, temp_hit) && temp_hit.t < hit.t)
				{
					hit         = temp_hit;
					hit.tri_idx = tri_idx;
					found       = true;
				}
			}
		}
		else
		{
			for (Uint32 i = 0; i < node.child_count; i++)
			{
				IntersectRecursiveImpl(bvh, triangles, ray, node.children[i], hit, found);
			}
		}
	}

	Bool IntersectRecursive(BLAS const& blas, Ray const& ray, HitInfo& hit)
	{
		if (blas.bvh.nodes.empty())
		{
			return false;
		}
		Bool found = false;
		Ray ray_copy = ray;
		IntersectRecursiveImpl(blas.bvh, blas.triangles, ray_copy, 0, hit, found);
		return found;
	}

	Bool Intersect(BLAS const& blas, Ray& ray, HitInfo& hit)
	{
		return amber::Intersect(blas.bvh, blas.triangles, ray, hit);
	}

	void Refit(BLAS& blas)
	{
		for (Int i = static_cast<Int>(blas.bvh.nodes_used) - 1; i >= 0; i--)
		{
			BVH8Node& node = blas.bvh.nodes[i];
			if (node.IsLeaf())
			{
				node.aabb_min = Vector3(BVH_INFINITY, BVH_INFINITY, BVH_INFINITY);
				node.aabb_max = Vector3(-BVH_INFINITY, -BVH_INFINITY, -BVH_INFINITY);
				for (Uint32 j = 0; j < node.prim_count; j++)
				{
					Uint32 tri_idx = blas.bvh.prim_indices[node.first_prim + j];
					Triangle const& tri = blas.triangles[tri_idx];
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
					BVH8Node const& child = blas.bvh.nodes[node.children[j]];
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
