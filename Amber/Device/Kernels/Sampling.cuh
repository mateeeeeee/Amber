#pragma once 

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

__device__ float3 OffsetRay(float3 p, float3 n)
{
	/* A Fast and Robust Method for Avoiding
	Self-Intersection by Carsten Wächter and Nikolaus Binder
	Chapter 6. Ray Tracing Gems NVIDIA */

	const float origin = 1.0f / 32.0f;
	const float floatScale = 1.0f / 65536.0f;
	const float intScale = 256.0f;

	int3 of_i = make_int3(intScale * n.x, intScale * n.y, intScale * n.z);

	int px_int = __float_as_int(p.x);
	int py_int = __float_as_int(p.y);
	int pz_int = __float_as_int(p.z);

	int3 adjusted_int = make_int3(
		(p.x < 0.0f) ? (px_int - of_i.x) : (px_int + of_i.x),
		(p.y < 0.0f) ? (py_int - of_i.y) : (py_int + of_i.y),
		(p.z < 0.0f) ? (pz_int - of_i.z) : (pz_int + of_i.z)
	);

	float3 p_i = make_float3(
		__int_as_float(adjusted_int.x),
		__int_as_float(adjusted_int.y),
		__int_as_float(adjusted_int.z)
	);

	return make_float3(
		(fabsf(p.x) < origin) ? (p.x + floatScale * n.x) : p_i.x,
		(fabsf(p.y) < origin) ? (p.y + floatScale * n.y) : p_i.y,
		(fabsf(p.z) < origin) ? (p.z + floatScale * n.z) : p_i.z
	);
}
