#pragma once
#include "Primitives.h"

namespace amber
{
	struct AABB
	{
		Vector3 min;
		Vector3 max;

		AABB() : min(1e30f, 1e30f, 1e30f), max(-1e30f, -1e30f, -1e30f) {}
		AABB(Vector3 const& min, Vector3 const& max) : min(min), max(max) {}

		void Grow(Vector3 const& point)
		{
			min.x = std::min(min.x, point.x);
			min.y = std::min(min.y, point.y);
			min.z = std::min(min.z, point.z);
			max.x = std::max(max.x, point.x);
			max.y = std::max(max.y, point.y);
			max.z = std::max(max.z, point.z);
		}

		void Grow(AABB const& other)
		{
			Grow(other.min);
			Grow(other.max);
		}

		Float Area() const
		{
			Vector3 extent = max - min;
			return extent.x * extent.y + extent.y * extent.z + extent.z * extent.x;
		}
	};

	// Slab test for ray-AABB intersection
	inline Bool IntersectAABB(Ray const& ray, Vector3 const& bmin, Vector3 const& bmax, Float t_max)
	{
		Float tx1 = (bmin.x - ray.origin.x) * ray.inv_direction.x;
		Float tx2 = (bmax.x - ray.origin.x) * ray.inv_direction.x;
		Float tmin = std::min(tx1, tx2);
		Float tmax = std::max(tx1, tx2);

		Float ty1 = (bmin.y - ray.origin.y) * ray.inv_direction.y;
		Float ty2 = (bmax.y - ray.origin.y) * ray.inv_direction.y;
		tmin = std::max(tmin, std::min(ty1, ty2));
		tmax = std::min(tmax, std::max(ty1, ty2));

		Float tz1 = (bmin.z - ray.origin.z) * ray.inv_direction.z;
		Float tz2 = (bmax.z - ray.origin.z) * ray.inv_direction.z;
		tmin = std::max(tmin, std::min(tz1, tz2));
		tmax = std::min(tmax, std::max(tz1, tz2));

		return tmax >= tmin && tmin < t_max && tmax > 0;
	}

	struct HitInfo
	{
		Float t = std::numeric_limits<Float>::max();
		Float u, v;
		Uint32 tri_idx;
	};

	// https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm
	inline Bool IntersectTriangle(Ray const& ray, Triangle const& tri, HitInfo& hit)
	{
		static constexpr Float EPSILON = 1e-8f;

		Vector3 edge1 = tri.v1 - tri.v0;
		Vector3 edge2 = tri.v2 - tri.v0;
		Vector3 h = Vector3::Cross(ray.direction, edge2);
		Float a = edge1.Dot(h);
		if (a > -EPSILON && a < EPSILON)
		{
			return false;
		}

		Float f = 1.0f / a;
		Vector3 s = ray.origin - tri.v0;
		Float u = f * s.Dot(h);
		if (u < 0.0f || u > 1.0f)
		{
			return false;
		}

		Vector3 q = Vector3::Cross(s, edge1);
		Float v = f * ray.direction.Dot(q);
		if (v < 0.0f || u + v > 1.0f)
		{
			return false;
		}

		Float t = f * edge2.Dot(q);
		if (t > EPSILON)
		{
			hit.t = t;
			hit.u = u;
			hit.v = v;
			return true;
		}
		return false;
	}
}
