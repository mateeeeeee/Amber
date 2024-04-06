#pragma once
#include <optix_device.h>
#include <vector_functions.h>
#include "OptixShared.h"

using namespace lavender;
extern "C" 
{
	__constant__ Params params;
}

__forceinline__ __device__ float3 toSRGB(const float3& c)
{
	float  invGamma = 1.0f / 2.2f;
	float3 powed = make_float3(powf(c.x, invGamma), powf(c.y, invGamma), powf(c.z, invGamma));
	return make_float3(
		c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
		c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
		c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f);
}

//__forceinline__ __device__ float dequantizeUnsigned8Bits( const unsigned char i )
//{
//    enum { N = (1 << 8) - 1 };
//    return min((float)i / (float)N), 1.f)
//}
__forceinline__ __device__ unsigned char quantizeUnsigned8Bits(float x)
{
	//x = clamp(x, 0.0f, 1.0f);

	enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
	return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}

__forceinline__ __device__ uchar4 make_color(const float3& c)
{
	// first apply gamma, then convert to unsigned char
	float3 srgb = toSRGB(c);
	return make_uchar4(quantizeUnsigned8Bits(srgb.x), quantizeUnsigned8Bits(srgb.y), quantizeUnsigned8Bits(srgb.z), 255u);
}


static __forceinline__ __device__ void setPayload(float3 p)
{
	optixSetPayload_0(__float_as_uint(p.x));
	optixSetPayload_1(__float_as_uint(p.y));
	optixSetPayload_2(__float_as_uint(p.z));
}


static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction)
{
	//const float3 U = params.cam_u;
	//const float3 V = params.cam_v;
	//const float3 W = params.cam_w;
	//const float2 d = make_float2(
	//	static_cast<float>(idx.x) / static_cast<float>(dim.x),
	//	static_cast<float>(idx.y) / static_cast<float>(dim.y)
	//);
	//
	//origin = params.cam_eye;
}


extern "C" __global__ void RG_NAME(rg)()
{
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	float3 ray_origin    = make_float3(0.0f, 0.0f, -1.0f);
	float3 ray_direction = make_float3(0.0f, 0.0f,  1.0f);
	
	//computeRay(idx, dim, ray_origin, ray_direction);
	unsigned int p0, p1, p2;
	p0 = __float_as_uint(1.0f);
	p1 = __float_as_uint(0.0f);
	p2 = __float_as_uint(0.0f);
	//setPayload(make_float3(1, 0, 0));
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
	float3 result;
	result.x = __uint_as_float(p0);
	result.y = __uint_as_float(p1);
	result.z = __uint_as_float(p2);
	params.image[idx.y * params.image_width + idx.x] = make_color(result);
}


extern "C" __global__ void __miss__ms()
{
	MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
	setPayload(make_float3(0.0f, 0.0f, 1.0f));
}


extern "C" __global__ void __closesthit__ch()
{
	//const float2 barycentrics = optixGetTriangleBarycentrics();
	setPayload(make_float3(0.0f, 1.0f, 0.0f));
}

