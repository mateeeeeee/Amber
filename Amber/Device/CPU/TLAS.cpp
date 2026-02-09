#include "TLAS.h"
#include "Utilities/Stack.h"

namespace amber
{
	Bool Intersect(BLASInstance const& instance, Uint32 instance_idx, Ray& ray, HitInfo& hit)
	{
		if (IntersectAABB(ray, instance.world_bounds.min, instance.world_bounds.max) == BVH_INFINITY)
		{
			return false;
		}

		Ray local_ray{};
		local_ray.origin    = Vector3::Transform(ray.origin, instance.inv_transform);
		local_ray.direction = TransformDirection(ray.direction, instance.inv_transform);
		local_ray.inv_direction.x = 1.0f / local_ray.direction.x;
		local_ray.inv_direction.y = 1.0f / local_ray.direction.y;
		local_ray.inv_direction.z = 1.0f / local_ray.direction.z;
		local_ray.t = ray.t;

		Bool found = amber::Intersect(instance.blas->bvh, local_ray, hit);
		if (found)
		{
			ray.t            = hit.t;
			hit.instance_idx = instance_idx;
			hit.instance_id  = instance.instance_id;
		}
		return found;
	}

	Bool Intersect(TLAS const& tlas, Ray& ray, HitInfo& hit)
	{
		if (tlas.instances.empty())
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
					Uint32 instance_idx = tlas.bvh.tri_indices[node->left_first + i];
					if (amber::Intersect(tlas.instances[instance_idx], instance_idx, ray, hit))
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
