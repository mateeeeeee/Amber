#include <functional>
#include "BLAS.h"
#include "BVH/Traversal.h"
#include "BVH/Refit.h"

namespace amber
{
	Bool IntersectRecursive(BLAS const& blas, Ray const& ray, HitInfo& hit)
	{
		if (blas.bvh.nodes.empty()) 
		{
			return false;
		}

		BVH8Node const& root = blas.bvh.nodes[0];
		if (IntersectAABB(ray, root.aabb_min, root.aabb_max) == BVH_INFINITY) 
		{
			return false;
		}

		Bool found = false;
		std::function<void(Uint32)> recurse = [&](Uint32 node_idx)
		{
			BVH8Node const& node = blas.bvh.nodes[node_idx];
			if (IntersectAABB(ray, node.aabb_min, node.aabb_max) == BVH_INFINITY) 
			{
				return;
			}
			if (node.IsLeaf())
			{
				for (Uint32 i = 0; i < node.prim_count; i++)
				{
					Uint32 tri_idx = blas.bvh.prim_indices[node.first_prim + i];
					HitInfo temp_hit;
					if (IntersectTriangle(ray, blas.triangles[tri_idx], temp_hit) && temp_hit.t < hit.t)
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
					recurse(node.children[i]);
				}
			}
		};
		recurse(0);
		return found;
	}

	Bool Intersect(BLAS const& blas, Ray& ray, HitInfo& hit)
	{
		return Intersect(blas.bvh, blas.triangles, ray, hit);
	}

	void Refit(BLAS& blas)
	{
		Refit(blas.bvh, blas.triangles);
	}
}
