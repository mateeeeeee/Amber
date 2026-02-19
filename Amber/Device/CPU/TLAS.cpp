#include "TLAS.h"
#include "BVH/Traversal.h"
#include "Utilities/Stack.h"
#include <algorithm>

namespace amber
{
	Bool Intersect(TLAS const& tlas, Ray& ray, HitInfo& hit)
	{
		if (tlas.bvh.nodes.empty())
		{
			return false;
		}

		BVH8Node const* node = &tlas.bvh.nodes[0];
		SmallStack<BVH8Node const*, 64 * 8> stack;
		Bool found = false;

		while (true)
		{
			if (node->IsLeaf())
			{
				for (Uint32 i = 0; i < node->prim_count; i++)
				{
					Uint32 instance_idx = tlas.bvh.prim_indices[node->first_prim + i];
					BLASInstance const& instance = tlas.instances[instance_idx];

					if (IntersectAABB(ray, instance.world_bounds.min, instance.world_bounds.max) == BVH_INFINITY)
					{
						continue;
					}

					Ray local_ray{};
					local_ray.origin        = Vector3::Transform(ray.origin, instance.inv_transform);
					local_ray.direction     = TransformDirection(ray.direction, instance.inv_transform);
					local_ray.inv_direction.x = 1.0f / local_ray.direction.x;
					local_ray.inv_direction.y = 1.0f / local_ray.direction.y;
					local_ray.inv_direction.z = 1.0f / local_ray.direction.z;
					local_ray.t             = ray.t;
					local_ray.flags         = ray.flags;

					if (amber::Intersect(instance.blas->bvh, instance.blas->triangles, local_ray, hit))
					{
						ray.t            = local_ray.t;
						hit.instance_idx = instance_idx;
						hit.instance_id  = instance.instance_id;
						found            = true;
						if (Bool(ray.flags & RayFlags::AcceptFirstHit))
						{
							return true;
						}
					}
				}
				if (stack.IsEmpty()) break;
				node = stack.Pop();
				continue;
			}

			struct ChildHit { BVH8Node const* node; Float dist; };
			ChildHit hits[8];
			Uint32 hit_count = 0;
			for (Uint32 i = 0; i < node->child_count; i++)
			{
				BVH8Node const* child = &tlas.bvh.nodes[node->children[i]];
				Float dist = IntersectAABB(ray, child->aabb_min, child->aabb_max);
				if (dist != BVH_INFINITY)
				{
					hits[hit_count++] = { child, dist };
				}
			}

			if (hit_count == 0)
			{
				if (stack.IsEmpty()) break;
				node = stack.Pop();
				continue;
			}

			std::sort(hits, hits + hit_count, [](ChildHit const& a, ChildHit const& b)
			{
				return a.dist > b.dist;
			});

			node = hits[hit_count - 1].node;
			for (Uint32 i = 0; i < hit_count - 1; i++)
			{
				stack.Push(hits[i].node);
			}
		}

		return found;
	}
}
