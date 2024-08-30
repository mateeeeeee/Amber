#pragma once
#include <vector_types.h>
#include <vector_functions.h>
#include "CudaMath.cuh"

//misc utilities

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


__device__ __forceinline__ float3 ToSRGB(float3 const& color)
{
	static constexpr float INV_GAMMA = 1.0f / 2.2f;
	float3 gamma_corrected_color = make_float3(powf(color.x, INV_GAMMA), powf(color.y, INV_GAMMA), powf(color.z, INV_GAMMA));
	return make_float3(
		color.x < 0.0031308f ? 12.92f * color.x : 1.055f * gamma_corrected_color.x - 0.055f,
		color.y < 0.0031308f ? 12.92f * color.y : 1.055f * gamma_corrected_color.y - 0.055f,
		color.z < 0.0031308f ? 12.92f * color.z : 1.055f * gamma_corrected_color.z - 0.055f);
}
__device__ __forceinline__ unsigned char QuantizeUnsigned8Bits(float x)
{
	x = clamp(x, 0.0f, 1.0f);
	static constexpr unsigned int N = (1 << 8) - 1;
	static constexpr unsigned int Np1 = (1 << 8);
	return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}
__device__ __forceinline__ uchar4 MakeColor(float3 const& c)
{
	float3 srgb = ToSRGB(c);
	return make_uchar4(QuantizeUnsigned8Bits(srgb.x), QuantizeUnsigned8Bits(srgb.y), QuantizeUnsigned8Bits(srgb.z), 255u);
}

template<typename T>
__forceinline__ __device__ T Interpolate(T const& t0, T const& t1, T const& t2, float2 bary)
{
	return t0 * (1.0f - bary.x - bary.y) + bary.x * t1 + bary.y * t2;
}

__forceinline__ __device__ void CosineSampleHemisphere(float u1, float u2, float3& p)
{
	const float r = sqrtf(u1);
	const float phi = 2.0f * M_PI * u2;
	p.x = r * cosf(phi);
	p.y = r * sinf(phi);
	p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

#endif