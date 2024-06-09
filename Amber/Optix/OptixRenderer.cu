#pragma once
#include <optix.h>
#include "OptixShared.h"
#include "CudaMath.h"
#include "CudaRandom.h"

using namespace amber;
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
template<typename T>
__forceinline__ __device__ T const& GetShaderParams()
{
	return *reinterpret_cast<T const*>(optixGetSbtDataPointer());
}


__device__ 
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
		0,
		0,
		p0, p1, p2);
	payload.radiance = make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
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

	unsigned int seed = tea<4>(pixel.y * screen.x + pixel.x, samples + params.sample_count);
	float3 result = make_float3(0.0f);
	do
	{
		float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));
		float2 d = (make_float2(pixel) + subpixel_jitter) / make_float2(screen);
		d = 2.0f * d - 1.0f;
		const float tan_half_fovy = tan(params.cam_fovy * 0.5f);
		const float aspect_ratio = params.cam_aspect_ratio;

		float3 ray_direction = normalize(d.x * aspect_ratio * tan_half_fovy * U + d.y * tan_half_fovy * V + W);
		float3 ray_origin = eye;

		Payload p{};
		p.attenuation = make_float3(1.0f);
		p.seed = seed;
		p.depth = 0;

		TraceRadiance(scene, ray_origin, ray_direction, 1e-5f, 1e16f, p);

		result += p.radiance;
	} while (--samples);

	result = result / params.sample_count;
	params.image[pixel.x + pixel.y * screen.x] = MakeColor(result);
}

#define M_PIF 3.14159265358979323846f
#define M_1_PIF 0.318309886183790671538f

extern "C" __global__ void MISS_NAME(ms)()
{
	float3 dir = optixGetWorldRayDirection();
	float u = (1.f + atan2(dir.x, -dir.z) * M_1_PIF) * 0.5f;
	float v = 1.0f - acos(dir.y) * M_1_PIF;

	if (params.sky)
	{
		float4 sampled = tex2D<float4>(params.sky, u, v);
		SetPayload(make_float3(sampled));
	}
	else
	{
		MissData const& miss_data = GetShaderParams<MissData>();
		SetPayload(miss_data.bg_color);
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
	unsigned int i0 = primitive_indices.x;
	unsigned int i1 = primitive_indices.y;
	unsigned int i2 = primitive_indices.z;

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


extern "C" __global__ void AH_NAME(ah)()
{
	unsigned int instance_idx = optixGetInstanceIndex();
	unsigned int primitive_idx = optixGetPrimitiveIndex();

	MeshGPU mesh = params.meshes[instance_idx];
	VertexData vertex = LoadVertexData(mesh, optixGetPrimitiveIndex(), optixGetTriangleBarycentrics());
	MaterialGPU material = params.materials[mesh.material_idx];

	if (material.diffuse_tex_id >= 0)
	{
		float4 sampled = tex2D<float4>(params.textures[material.diffuse_tex_id], vertex.uv.x, vertex.uv.y);
		if(sampled.w < 0.5f) optixIgnoreIntersection();
	}
}

extern "C" __global__ void CH_NAME(ch)()
{
	unsigned int instance_idx = optixGetInstanceIndex();
	unsigned int primitive_idx = optixGetPrimitiveIndex();

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
	SetPayload(result);
}

