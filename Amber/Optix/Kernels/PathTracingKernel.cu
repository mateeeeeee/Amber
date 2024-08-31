#pragma once
#include <stdio.h>
#include <optix.h>
#include "CudaRandom.cuh"
#include "CudaUtils.cuh"
#include "Optix/OptixShared.h"

using namespace amber;

extern "C" 
{
	__constant__ LaunchParams params;
}

__device__ inline uint32 PackPointer0(void* ptr) 
{
	uintptr uptr = reinterpret_cast<uintptr>(ptr);
	return static_cast<uint32>(uptr >> 32);
}
__device__ inline uint32 PackPointer1(void* ptr) 
{
	uintptr uptr = reinterpret_cast<uintptr>(ptr);
	return static_cast<uint32>(uptr);
}

template <typename T>
__device__ __forceinline__ T* GetPayload()
{
    uint32 p0 = optixGetPayload_0(), p1 = optixGetPayload_1();
    const uintptr uptr = (uintptr(p0) << 32) | p1;
    return reinterpret_cast<T *>(uptr);
}

template <typename... Args>
__device__ __forceinline__ void Trace(
	OptixTraversableHandle traversable,
	float3 ray_origin, 
	float3 ray_direction,
	float tmin,
	float tmax, Args&&... payload)
{
	optixTrace(traversable, ray_origin, ray_direction, 
		tmin, tmax, 0.0f,
		OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0,
		0, 
		0,
		std::forward<Args>(payload)...);
}

__device__ __forceinline__ bool TraceOcclusion(
	OptixTraversableHandle handle,
	float3                 ray_origin,
	float3                 ray_direction,
	float                  tmin,
	float                  tmax
)
{
	optixTraverse(
		handle,
		ray_origin,
		ray_direction,
		tmin,
		tmax, 0.0f,                
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0,                          
		1,							
		0                           
	);
	return optixHitObjectIsHit();
}

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
	OptixTraversableHandle scene = params.traversable;
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

		RadiancePRD prd{};
		prd.attenuation = make_float3(1.0f);
		prd.seed = seed;
		prd.depth = 0;
		
		uint32 p0 = PackPointer0(&prd), p1 = PackPointer1(&prd);
		for (uint32 bounce = 0; bounce < 1; ++bounce)
		{
			Trace(scene, ray_origin, ray_direction, 1e-5f, 1e16f, p0, p1);

			final_color += prd.emissive;
			final_color += prd.radiance * prd.attenuation;

			const float p = dot(prd.attenuation, make_float3(0.30f, 0.59f, 0.11f));
			const bool done = prd.done || rnd(prd.seed) > p;
			if (done) break;

			prd.attenuation /= p;
			prd.depth++;
			ray_origin = prd.origin;
			ray_direction = prd.direction;
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

	RadiancePRD* prd = GetPayload<RadiancePRD>();
	prd->radiance = make_float3(0.0f);
	if (params.sky)
	{
		float4 sampled = tex2D<float4>(params.sky, u, v);
		prd->radiance = make_float3(sampled.x, sampled.y, sampled.z);
	}
	prd->emissive = make_float3(0.0f);
	prd->done = true;
}

struct VertexData
{
	float3 P;
	float3 N;
	float2 uv;
};

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
	vertex.P = Interpolate(pos0, pos1, pos2, barycentrics);

	float3* mesh_normals = params.normals + mesh.normals_offset;
	float3 nor0 = mesh_normals[i0];
	float3 nor1 = mesh_normals[i1];
	float3 nor2 = mesh_normals[i2];
	vertex.N = Interpolate(nor0, nor1, nor2, barycentrics);
	
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


//__device__ float3 SampleDirectLight(MaterialGPU const& material,
//	float3 const& P,
//	float3 const& N,
//	float3 const& v_x,
//	float3 const& v_y,
//	float3 const& w_o)
//{
//	float3 illum = make_float3(0.f);
//
//	//later, pick one light randomly and accumulate result
//
//	uint32 occlusion_flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT |
//							 OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
//							 OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
//
//	for (uint32 i = 0; i < params.light_count; ++i)
//	{
//		LightGPU light = params.lights[i];
//		if (light.type == LightType_Directional)
//		{
//
//		}
//	}
//}


extern "C" 
__global__ void CH_NAME(ch)()
{
	uint32 instance_idx = optixGetInstanceIndex();
	uint32 primitive_idx = optixGetPrimitiveIndex();

	MeshGPU mesh = params.meshes[instance_idx];
	VertexData vertex = LoadVertexData(mesh, optixGetPrimitiveIndex(), optixGetTriangleBarycentrics());
	MaterialGPU material = params.materials[mesh.material_idx];

	RadiancePRD* prd = GetPayload<RadiancePRD>();
	if (material.diffuse_tex_id >= 0)
	{
		float4 sampled = tex2D<float4>(params.textures[material.diffuse_tex_id], vertex.uv.x, vertex.uv.y);
		prd->radiance = material.base_color * make_float3(sampled.x, sampled.y, sampled.z);
	}
	else
	{
		prd->radiance = material.base_color;
	}

	if (prd->depth == 0)
	{
		if (material.emissive_tex_id >= 0)
		{
			float4 sampled = tex2D<float4>(params.textures[material.emissive_tex_id], vertex.uv.x, vertex.uv.y);
			prd->emissive = material.emissive_color * make_float3(sampled.x, sampled.y, sampled.z);
		}
		else
		{
			prd->emissive = material.emissive_color;
		}
	}

	LightGPU light = params.lights[0];
	if (TraceOcclusion(params.traversable, vertex.P, -light.direction, 0.001f, 100000.0f))
	{
		prd->radiance = make_float3(0.0f);
	}

	//OrthonormalBasis onb(vertex.N);
	//float3 n = onb.normal;
	//float3 t = onb.tangent;
	//float3 b = onb.binormal;
	//const float3 w_o = -prd->direction;


	//prd->radiance += prd->attenuation * SampleDirectLight(mat,
	//	hit_p,
	//	v_z,
	//	v_x,
	//	v_y,
	//	w_o,
	//	params.lights,
	//	params.num_lights,
	//	ray_count,
	//	rng);

	float z1 = rnd(prd->seed);
	float z2 = rnd(prd->seed);

	float3 w_in;
	CosineSampleHemisphere(z1, z2, w_in);
	OrthonormalBasis onb(vertex.N);
	onb.InverseTransform(w_in);
	prd->direction = w_in;
	prd->origin = vertex.P;
	prd->done = false;
}

