#pragma once 

#if defined(__CUDACC__)

struct OrthonormalBasis
{
	__forceinline__ __device__ OrthonormalBasis(float3 const& _normal)
	{
		normal = _normal;

		if (fabs(normal.x) > fabs(normal.z))
		{
			binormal.x = -normal.y;
			binormal.y = normal.x;
			binormal.z = 0;
		}
		else
		{
			binormal.x = 0;
			binormal.y = -normal.z;
			binormal.z = normal.y;
		}

		binormal = normalize(binormal);
		tangent = cross(binormal, normal);
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

#endif