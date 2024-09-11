#pragma once
#include <vector_types.h>
#include <vector_functions.h>
#include "Math.cuh"


#if defined(__CUDACC__)

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

#endif