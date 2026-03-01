#pragma once
#include "Sampler.h"
#include "TLAS.h"
#include "Scene/Light.h"
<<<<<<< HEAD
=======
#include "Math/MathCommon.h"
>>>>>>> bvh-benchmark
#include <cmath>
#include <vector>

namespace amber
{
	inline Uint32 PcgHash(Uint32 v)
	{
		v = v * 747796405u + 2891336453u;
		v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
		return (v >> 22u) ^ v;
	}

	inline Float RandFloat(Uint32& rng)
	{
		rng = PcgHash(rng);
		return (rng >> 8) * (1.0f / 16777216.0f);
	}

	inline Vector3 CosineSampleHemisphere(Vector3 const& normal, Float r1, Float r2)
	{
		Vector3 up        = std::abs(normal.x) < 0.9f ? Vector3(1, 0, 0) : Vector3(0, 1, 0);
		Vector3 tangent   = Vector3::Cross(up, normal).Normalized();
		Vector3 bitangent = Vector3::Cross(normal, tangent);

		Float r   = std::sqrt(r1);
<<<<<<< HEAD
		Float phi = 2.0f * M_PI * r2;
=======
		Float phi = 2.0f * amber::PI * r2;
>>>>>>> bvh-benchmark
		Float x   = r * std::cos(phi);
		Float y   = r * std::sin(phi);
		Float z   = std::sqrt(std::max(0.0f, 1.0f - r1));

		return (tangent * x + bitangent * y + normal * z).Normalized();
	}

	inline Vector3 SampleEnvironment(Texture const& env, Vector3 const& dir)
	{
		if (!env.data)
		{
			return Vector3(25.0f / 255.0f, 25.0f / 255.0f, 25.0f / 255.0f);
		}
<<<<<<< HEAD
		Float u = (1.0f + std::atan2(dir.x, -dir.z) * M_1_PI) * 0.5f;
		Float v = /*1.0f -*/ std::acos(dir.y) * M_1_PI;
=======
		Float u = (1.0f + std::atan2(dir.x, -dir.z) * amber::INV_PI) * 0.5f;
		Float v = /*1.0f -*/ std::acos(dir.y) * amber::INV_PI;
>>>>>>> bvh-benchmark
		return BilinearClamp.Sample<Vector3>(env, Vector2(u, v));
	}

	inline Vector3 TonemapReinhard(Vector3 c)
	{
		c.x = c.x / (1.0f + c.x);
		c.y = c.y / (1.0f + c.y);
		c.z = c.z / (1.0f + c.z);
		return c;
	}

	inline RGBA8 ToDisplay(Vector3 c, Float exposure = 1.0f, Int tonemap_mode = 1)
	{
		c = c * exposure;
		if (tonemap_mode == 1) c = TonemapReinhard(c);
		c.x = std::pow(std::clamp(c.x, 0.0f, 1.0f), 1.0f / 2.2f);
		c.y = std::pow(std::clamp(c.y, 0.0f, 1.0f), 1.0f / 2.2f);
		c.z = std::pow(std::clamp(c.z, 0.0f, 1.0f), 1.0f / 2.2f);
		return RGBA8::FromFloat(c.x, c.y, c.z);
	}

	inline Vector3 SampleDirectLight(TLAS const& tlas, std::vector<Light> const& lights, Vector3 const& hit_pos, Vector3 const& normal, Uint32& rng)
	{
		if (lights.empty()) 
		{
			return Vector3(0.0f, 0.0f, 0.0f);
		}

		Uint32 light_idx = static_cast<Uint32>(RandFloat(rng) * lights.size()) % static_cast<Uint32>(lights.size());
		Light const& light = lights[light_idx];
		Float light_count = static_cast<Float>(lights.size());

		Vector3 to_light;
		Float   max_t;
		if (light.type == LightType::Directional)
		{
			to_light = (-light.direction).Normalized();
			max_t    = BVH_INFINITY;
		}
		else if (light.type == LightType::Point)
		{
			Vector3 diff = light.position - hit_pos;
			Float   dist = std::sqrt(diff.Dot(diff));
			to_light = diff * (1.0f / dist);
			max_t    = dist - 1e-3f;
		}
		else
		{
			return Vector3(0.0f, 0.0f, 0.0f);
		}

		Float n_dot_l = normal.Dot(to_light);
		if (n_dot_l <= 0.0f) return Vector3(0.0f, 0.0f, 0.0f);

		Ray shadow_ray(hit_pos, to_light, RayFlags::AcceptFirstHit);
		shadow_ray.t = max_t;
		HitInfo shadow_hit;
		if (Intersect(tlas, shadow_ray, shadow_hit))
		{
			return Vector3(0.0f, 0.0f, 0.0f);
		}

		Vector3 radiance = light.color;
		if (light.type == LightType::Point)
		{
			Vector3 diff = light.position - hit_pos;
			Float dist2 = diff.Dot(diff);
			radiance = radiance * (1.0f / std::max(dist2, 1e-4f));
		}
		return radiance * (n_dot_l * light_count);
	}
}
