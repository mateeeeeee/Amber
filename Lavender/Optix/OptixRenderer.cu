#pragma once
#include <optix.h>
#include "OptixShared.h"
#include "CudaMath.h"

using namespace lavender;

extern "C" 
{
	__constant__ Params params;
}

__forceinline__ __device__ float3 ToSRGB(float3 const& color)
{
	static constexpr float INV_GAMMA = 1.0f / 2.2f;
	float3 gamma_corrected_color = make_float3(powf(color.x, INV_GAMMA), powf(color.y, INV_GAMMA), powf(color.z, INV_GAMMA));
	return make_float3(
		color.x < 0.0031308f ? 12.92f * color.x : 1.055f * gamma_corrected_color.x - 0.055f,
		color.y < 0.0031308f ? 12.92f * color.y : 1.055f * gamma_corrected_color.y - 0.055f,
		color.z < 0.0031308f ? 12.92f * color.z : 1.055f * gamma_corrected_color.z - 0.055f);
}
__forceinline__ __device__ unsigned char QuantizeUnsigned8Bits(float x)
{
	x = clamp(x, 0.0f, 1.0f);
	enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
	return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}
__forceinline__ __device__ uchar4 MakeColor(const float3& c)
{
	float3 srgb = ToSRGB(c);
	return make_uchar4(QuantizeUnsigned8Bits(srgb.x), QuantizeUnsigned8Bits(srgb.y), QuantizeUnsigned8Bits(srgb.z), 255u);
}

__forceinline__ __device__ void SetPayload(float3 p)
{
	optixSetPayload_0(__float_as_uint(p.x));
	optixSetPayload_1(__float_as_uint(p.y));
	optixSetPayload_2(__float_as_uint(p.z));
}
__forceinline__ __device__ float3 GetPayload(unsigned int p0, unsigned int p1, unsigned int p2)
{
	float3 p;
	p.x = __uint_as_float(p0);
	p.y = __uint_as_float(p1);
	p.z = __uint_as_float(p2);
	return p;
}

static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction)
{
	const float3 U = params.cam_u;
	const float3 V = params.cam_v;
	const float3 W = params.cam_w;
	const float2 d = 2.0f * make_float2(
		static_cast<float>(idx.x) / static_cast<float>(dim.x),
		static_cast<float>(idx.y) / static_cast<float>(dim.y)
	) - 1.0f;

	origin = params.cam_eye;
	direction = normalize(d.x * U + d.y * V + W);
}



extern "C" __global__ void RG_NAME(rg)()
{
	
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	// Map our launch idx to a screen location and create a ray from the camera
	// location through the screen
	float3 ray_origin, ray_direction;
	computeRay(idx, dim, ray_origin, ray_direction);

	unsigned int p0, p1, p2;
	optixTrace(
		params.handle,
		ray_origin,
		ray_direction,
		0.0f,						
		1e16f,						
		0.0f,						
		OptixVisibilityMask(255),	
		OPTIX_RAY_FLAG_NONE,
		0,                   
		1,                   
		0,                   
		p0, p1, p2);
	float3 result = GetPayload(p0, p1, p2);
	params.image[(params.image_height - idx.y - 1) * params.image_width + idx.x] = MakeColor(result);
}


extern "C" __global__ void __miss__ms()
{
	MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
	SetPayload(make_float3(0.0f, 0.0f, 1.0f));
}


extern "C" __global__ void __closesthit__ch()
{
	//const float2 barycentrics = optixGetTriangleBarycentrics();
	SetPayload(make_float3(0.0f, 1.0f, 0.0f));
}

