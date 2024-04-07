#pragma once
#include <optix.h>
#include "OptixShared.h"
#include "CudaMath.h"
#include "CudaRandom.h"

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

struct Payload
{
	float3 color;
};

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
template<typename T>
__forceinline__ __device__ T const& GetShaderParams()
{
	return *reinterpret_cast<T const*>(optixGetSbtDataPointer());
}


__forceinline__ __device__ 
void TraceRadiance(OptixTraversableHandle scene,
	float3                 rayOrigin,
	float3                 rayDirection,
	float                  tmin,
	float                  tmax,
	Payload& payload)
{
	unsigned int p0, p1, p2;
	optixTrace(
		scene,
		rayOrigin,
		rayDirection,
		tmin,
		tmax,
		0.0f,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_NONE,
		0,
		1,
		0,
		p0, p1, p2);
	payload.color = make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
}


extern "C" __global__ void RG_NAME(rg)()
{
	OptixTraversableHandle scene = params.handle;
	float3 const  eye = params.cam_eye;
	float3 const  U = params.cam_u;
	float3 const  V = params.cam_v;
	float3 const  W = params.cam_w;
	uint2  const  pixel  = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
	uint2  const  screen = make_uint2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y);
	
	int samples = params.sample_count;
	float3 result = make_float3(0.0f);
	do
	{
		unsigned int seed = tea<4>(pixel.y * screen.x + pixel.x, samples + params.frame_index * params.sample_count);
		float2 subpixelJitter = make_float2(rnd(seed), rnd(seed));
		float2 d = (make_float2(pixel) + subpixelJitter) / make_float2(screen);
		d = 2.0f * d - 1.0f;

		const float tanFovyHalf = tan(params.cam_fovy * 0.5f);
		const float aspectRatio = params.cam_aspect_ratio;

		float3 rayDirection = normalize(d.x * aspectRatio * tanFovyHalf * U + d.y * tanFovyHalf * V + W);
		float3 rayOrigin = eye;

		Payload p{};
		TraceRadiance(scene, rayOrigin, rayDirection, 1e-5f, 1e16f, p);
		result += p.color;
	} while (--samples);

	result = result / params.sample_count;
	params.image[pixel.x + pixel.y * screen.x] = MakeColor(result);
}


extern "C" __global__ void __miss__ms()
{
	MissData const& miss_data = GetShaderParams<MissData>(); 
	SetPayload(make_float3(0.0f, 0.0f, 1.0f));
}


extern "C" __global__ void __closesthit__ch()
{
	const float2 barycentrics = optixGetTriangleBarycentrics();
	float4 sampled = tex2D<float4>(params.textures[0], barycentrics.x, barycentrics.y);
	SetPayload(make_float3(sampled));
}

