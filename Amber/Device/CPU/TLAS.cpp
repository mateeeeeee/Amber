#include "TLAS.h"
#include "Utilities/Stack.h"

namespace amber
{
	Bool Intersect(TLAS const& tlas, Ray& ray, HitInfo& hit)
	{
		if (tlas.blas_count == 0)
		{
			return false;
		}

		BVHNode const* node = &tlas.bvh.nodes[0];
		SmallStack<BVHNode const*, 64> stack;
		Bool found = false;

		while (true)
		{
			if (node->IsLeaf())
			{
				for (Uint32 i = 0; i < node->tri_count; i++)
				{
					Uint32 blas_idx = tlas.bvh.tri_indices[node->left_first + i];
					if (amber::Intersect(tlas.blas_list[blas_idx], blas_idx, ray, hit))
					{
						found = true;
					}
				}
				if (stack.IsEmpty()) break;
				node = stack.Pop();
				continue;
			}

			BVHNode const* child1 = &tlas.bvh.nodes[node->left_first];
			BVHNode const* child2 = &tlas.bvh.nodes[node->left_first + 1];
			Float dist1 = IntersectAABB(ray, child1->aabb_min, child1->aabb_max);
			Float dist2 = IntersectAABB(ray, child2->aabb_min, child2->aabb_max);

			if (dist1 > dist2)
			{
				std::swap(dist1, dist2);
				std::swap(child1, child2);
			}

			if (dist1 == BVH_INFINITY)
			{
				if (stack.IsEmpty()) break;
				node = stack.Pop();
			}
			else
			{
				node = child1;
				if (dist2 != BVH_INFINITY) stack.Push(child2);
			}
		}

		return found;
	}
}
