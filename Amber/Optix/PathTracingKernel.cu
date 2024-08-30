#pragma once
#include <optix.h>
#include "OptixShared.h"
#include "CudaMath.cuh"
#include "CudaRandom.cuh"

using namespace amber;

extern "C" 
{
	__constant__ Params params;
}

struct Address
{
	uint32 low;
	uint32 high;
};

__device__ inline Address DecomposeAddress(void* ptr)
{
	uintptr addr = reinterpret_cast<uintptr>(ptr);
	uint32 low = static_cast<uint32>(addr & 0xFFFFFFFF);
	uint32 high = static_cast<uint32>((addr >> 32) & 0xFFFFFFFF);
	Address result;
	result.low = low;
	result.high = high;
	return result;
}

template <typename T>
__device__ __forceinline__ T* GetPayload()
{
    uint32 p0 = optixGetPayload_0(), p1 = optixGetPayload_1();
    const uintptr uptr = (uintptr(p0) << 32) | p1;
    return reinterpret_cast<T *>(uptr);
}

template <typename... Args>
__device__ __forceinline__ void Trace(OptixTraversableHandle traversable,
	float3 ray_origin, float3 ray_direction,
	float tmin,float tmax, Args&&... payload)
{
	optixTrace(traversable, ray_origin, ray_direction, 
		tmin, tmax, 0.0f,
		OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0,
		0, //or 1?
		0,
		std::forward<Args>(payload)...);
}

//color utility

__device__ __forceinline__ float3 ToSRGB(float3 const& color)
{
	static constexpr float INV_GAMMA = 1.0f / 2.2f;
	float3 gamma_corrected_color = make_float3(powf(color.x, INV_GAMMA), powf(color.y, INV_GAMMA), powf(color.z, INV_GAMMA));
	return make_float3(
		color.x < 0.0031308f ? 12.92f * color.x : 1.055f * gamma_corrected_color.x - 0.055f,
		color.y < 0.0031308f ? 12.92f * color.y : 1.055f * gamma_corrected_color.y - 0.055f,
		color.z < 0.0031308f ? 12.92f * color.z : 1.055f * gamma_corrected_color.z - 0.055f);
}
__device__ __forceinline__ uint8 QuantizeUnsigned8Bits(float x)
{
	x = clamp(x, 0.0f, 1.0f);
	static constexpr uint32 N = (1 << 8) - 1;
	static constexpr uint32 Np1 = (1 << 8);
	return (uint8)min((uint32)(x * (float)Np1), (uint32)N);
}
__device__ __forceinline__ uchar4 MakeColor(float3 const& c)
{
	float3 srgb = ToSRGB(c);
	return make_uchar4(QuantizeUnsigned8Bits(srgb.x), QuantizeUnsigned8Bits(srgb.y), QuantizeUnsigned8Bits(srgb.z), 255u);
}
//end color utility

__device__ __forceinline__ float3 GetRayDirection(uint2 pixel, uint2 screen, unsigned int seed)
{
	float3 const  U = params.cam_u;
	float3 const  V = params.cam_v;
	float3 const  W = params.cam_w;

	float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));
	float2 d = (make_float2(pixel) + subpixel_jitter) / make_float2(screen);
	d = 2.0f * d - 1.0f;
	float tan_half_fovy = tan(params.cam_fovy * 0.5f);
	float aspect_ratio = params.cam_aspect_ratio;
	float3 ray_direction = normalize(d.x * aspect_ratio * tan_half_fovy * U + d.y * tan_half_fovy * V + W);
	return ray_direction;
}


extern "C" 
__global__ void RG_NAME(rg)()
{
	OptixTraversableHandle scene = params.handle;
	float3 const  eye = params.cam_eye;
	uint2  const  pixel  = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
	uint2  const  screen = make_uint2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y);
	int samples = params.sample_count;

	float3 final_color = make_float3(0.0f);
	do
	{
		uint32 seed = tea<4>(pixel.y * screen.x + pixel.x, samples);
		float3 ray_origin = eye;
		float3 ray_direction = GetRayDirection(pixel, screen, seed);

		Payload p{};
		p.attenuation = make_float3(1.0f);
		p.seed = seed;
		
		Address addr = DecomposeAddress(&p);
		for (unsigned int bounce = 0; bounce < params.max_bounces; ++bounce)
		{
			Trace(scene, ray_origin, ray_direction, 1e-5f, 1e16f, addr.low, addr.high);
			final_color += p.attenuation * p.radiance;

			ray_origin = p.origin;
			ray_direction = p.direction;
		}
	} while (--samples);

	final_color = final_color / params.sample_count;
	params.image[pixel.x + pixel.y * screen.x] = MakeColor(final_color);
}

extern "C" 
__global__ void MISS_NAME(ms)()
{
	float3 dir = optixGetWorldRayDirection();
	float u = (1.f + atan2(dir.x, -dir.z) * M_INV_PI) * 0.5f;
	float v = 1.0f - acos(dir.y) * M_INV_PI;

	if (params.sky)
	{
		float4 sampled = tex2D<float4>(params.sky, u, v);
		GetPayload<Payload>()->radiance = make_float3(sampled.x, sampled.y, sampled.z);
	}
}

struct VertexData
{
	float3 pos;
	float3 nor;
	float2 uv;
};
template<typename T>
__forceinline__ __device__ T Interpolate(T const& t0, T const& t1, T const& t2, float2 bary)
{
	return t0 * (1.0f - bary.x - bary.y) + bary.x * t1 + bary.y * t2;
}
__device__ VertexData LoadVertexData(MeshGPU const& mesh, unsigned int primitive_idx, float2 barycentrics)
{
	VertexData vertex{};
	uint3* mesh_indices = params.indices + mesh.indices_offset;

	uint3 primitive_indices = mesh_indices[primitive_idx];
	uint32 i0 = primitive_indices.x;
	uint32 i1 = primitive_indices.y;
	uint32 i2 = primitive_indices.z;

	float3* mesh_vertices = params.vertices + mesh.positions_offset;
	float3 pos0 = mesh_vertices[i0];
	float3 pos1 = mesh_vertices[i1];
	float3 pos2 = mesh_vertices[i2];
	vertex.pos = Interpolate(pos0, pos1, pos2, barycentrics);

	float3* mesh_normals = params.normals + mesh.normals_offset;
	float3 nor0 = mesh_normals[i0];
	float3 nor1 = mesh_normals[i1];
	float3 nor2 = mesh_normals[i2];
	vertex.nor = Interpolate(nor0, nor1, nor2, barycentrics);
	
	float2* mesh_uvs = params.uvs + mesh.uvs_offset;
	float2 uv0 = mesh_uvs[i0];
	float2 uv1 = mesh_uvs[i1];
	float2 uv2 = mesh_uvs[i2];
	vertex.uv = Interpolate(uv0, uv1, uv2, barycentrics);
	return vertex;
}

extern "C" 
__global__ void AH_NAME(ah)()
{
	uint32 instance_idx = optixGetInstanceIndex();
	uint32 primitive_idx = optixGetPrimitiveIndex();

	MeshGPU mesh = params.meshes[instance_idx];
	VertexData vertex = LoadVertexData(mesh, optixGetPrimitiveIndex(), optixGetTriangleBarycentrics());
	MaterialGPU material = params.materials[mesh.material_idx];

	if (material.diffuse_tex_id >= 0)
	{
		float4 sampled = tex2D<float4>(params.textures[material.diffuse_tex_id], vertex.uv.x, vertex.uv.y);
		if(sampled.w < 0.5f) optixIgnoreIntersection();
	}
}
extern "C" 
__global__ void CH_NAME(ch)()
{
	uint32 instance_idx = optixGetInstanceIndex();
	uint32 primitive_idx = optixGetPrimitiveIndex();

	MeshGPU mesh = params.meshes[instance_idx];
	VertexData vertex = LoadVertexData(mesh, optixGetPrimitiveIndex(), optixGetTriangleBarycentrics());
	MaterialGPU material = params.materials[mesh.material_idx];
	float3 result{};
	if (material.diffuse_tex_id >= 0)
	{
		float4 sampled = tex2D<float4>(params.textures[material.diffuse_tex_id], vertex.uv.x, vertex.uv.y);
		result = make_float3(sampled);
	}
	else
	{
		result = material.base_color;
	}
	GetPayload<Payload>()->radiance = result;
}

