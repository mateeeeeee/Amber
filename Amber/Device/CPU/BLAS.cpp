#include "BLAS.h"

namespace amber
{
	BLAS::BLAS(BLAS const& other)
		: bvh(other.bvh), triangles(other.triangles),
		  transform(other.transform), inv_transform(other.inv_transform),
		  world_bounds(other.world_bounds)
	{
		bvh.triangles = &triangles;
	}

	BLAS::BLAS(BLAS&& other) noexcept
		: bvh(std::move(other.bvh)), triangles(std::move(other.triangles)),
		  transform(other.transform), inv_transform(other.inv_transform),
		  world_bounds(other.world_bounds)
	{
		bvh.triangles = &triangles;
	}

	BLAS& BLAS::operator=(BLAS const& other)
	{
		if (this != &other)
		{
			bvh = other.bvh;
			triangles = other.triangles;
			transform = other.transform;
			inv_transform = other.inv_transform;
			world_bounds = other.world_bounds;
			bvh.triangles = &triangles;
		}
		return *this;
	}

	BLAS& BLAS::operator=(BLAS&& other) noexcept
	{
		if (this != &other)
		{
			bvh = std::move(other.bvh);
			triangles = std::move(other.triangles);
			transform = other.transform;
			inv_transform = other.inv_transform;
			world_bounds = other.world_bounds;
			bvh.triangles = &triangles;
		}
		return *this;
	}

	void BLAS::SetTransform(Matrix const& t)
	{
		transform = t;
		inv_transform = t.Inverse();
		Vector3 bmin = bvh.nodes[0].aabb_min;
		Vector3 bmax = bvh.nodes[0].aabb_max;
		world_bounds = AABB();
		for (Int i = 0; i < 8; i++)
		{
			Vector3 corner(
				(i & 1) ? bmax.x : bmin.x,
				(i & 2) ? bmax.y : bmin.y,
				(i & 4) ? bmax.z : bmin.z
			);
			world_bounds.Grow(Vector3::Transform(corner, t));
		}
	}

	Bool Intersect(BLAS const& blas, Uint32 blas_idx, Ray& ray, HitInfo& hit)
	{
		if (IntersectAABB(ray, blas.world_bounds.min, blas.world_bounds.max) == BVH_INFINITY)
		{
			return false;
		}

		Ray local_ray{};
		local_ray.origin = Vector3::Transform(ray.origin, blas.inv_transform);
		local_ray.direction = TransformDirection(ray.direction, blas.inv_transform);
		local_ray.inv_direction.x = 1.0f / local_ray.direction.x;
		local_ray.inv_direction.y = 1.0f / local_ray.direction.y;
		local_ray.inv_direction.z = 1.0f / local_ray.direction.z;
		local_ray.t = ray.t;

		Bool found = amber::Intersect(blas.bvh, local_ray, hit);
		if (found)
		{
			ray.t = hit.t;
			hit.blas_idx = blas_idx;
		}
		return found;
	}
}
