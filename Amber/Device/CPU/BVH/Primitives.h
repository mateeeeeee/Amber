#pragma once

namespace amber
{
	inline constexpr Float BVH_INFINITY = 1e30f;
	inline constexpr Float BVH_EPSILON = 1e-8f;

	struct alignas(16) Ray
	{
		Vector3 origin;
		Vector3 direction;
		Vector3 inv_direction;
		Float t = BVH_INFINITY;

		Ray() = default;
		Ray(Vector3 const& origin, Vector3 const& direction)
			: origin(origin), direction(direction)
		{
			inv_direction.x = 1.0f / direction.x;
			inv_direction.y = 1.0f / direction.y;
			inv_direction.z = 1.0f / direction.z;
		}
	};

	struct alignas(16) Triangle
	{
		Vector3 v0, v1, v2;
		Vector3 centroid;
	};
}
