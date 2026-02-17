#pragma once
#include "Sampler.h"
#include <cmath>

namespace amber
{
	static constexpr Float PT_PI = 3.14159265359f;

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
		Float phi = 2.0f * PT_PI * r2;
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
		Float u = std::atan2(dir.z, dir.x) / (2.0f * PT_PI) + 0.5f;
		Float v = (std::asin(std::clamp(dir.y, -1.0f, 1.0f)) / PT_PI + 0.5f);
		return BilinearClamp.Sample<Vector3>(env, Vector2(u, v));
	}

	inline Vector3 TonemapReinhard(Vector3 c)
	{
		c.x = c.x / (1.0f + c.x);
		c.y = c.y / (1.0f + c.y);
		c.z = c.z / (1.0f + c.z);
		return c;
	}

	inline RGBA8 ToDisplay(Vector3 c)
	{
		c = TonemapReinhard(c);
		c.x = std::pow(c.x, 1.0f / 2.2f);
		c.y = std::pow(c.y, 1.0f / 2.2f);
		c.z = std::pow(c.z, 1.0f / 2.2f);
		return RGBA8::FromFloat(c.x, c.y, c.z);
	}
}
