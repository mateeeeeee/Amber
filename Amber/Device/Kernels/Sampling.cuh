#pragma once 
#include "Math.cuh"

struct OrthonormalBasis
{
	__forceinline__ __device__ OrthonormalBasis(float3 const& _normal)
	{
		normal = normalize(_normal); // Ensure the normal is normalized

		// Handle degenerate cases better
		if (fabs(normal.x) < 0.0001f && fabs(normal.y) < 0.0001f)
		{
			binormal = make_float3(1.0f, 0.0f, 0.0f);
		}
		else
		{
			binormal = make_float3(normal.y, -normal.x, 0.0f);
		}
		binormal = normalize(binormal);
		tangent = normalize(cross(binormal, normal));
	}

	__forceinline__ __device__ void InverseTransform(float3& p) const
	{
		p = p.x * tangent + p.y * binormal + p.z * normal;
	}

	float3 tangent;
	float3 binormal;
	float3 normal;
};

__forceinline__ __device__ void CosineSampleHemisphere(float u1, float u2, float3& p)
{
	const float r = sqrtf(u1);
	const float phi = 2.0f * M_PI * u2;
	p.x = r * cosf(phi);
	p.y = r * sinf(phi);
	p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}