#pragma once
#include <optix.h>
#include "CudaRandom.cuh"
#include "CudaUtils.cuh"
#include "Optix/OptixShared.h"

using namespace amber;

static constexpr uint32 RR_DEPTH = 2;

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

__device__ __forceinline__ float3 GetEmissive(uint32 material_idx, float2 uv)
{
	MaterialGPU material = params.materials[material_idx];
	if (material.emissive_tex_id >= 0)
	{
		float4 sampled = tex2D<float4>(params.textures[material.emissive_tex_id], uv.x, uv.y);
		return material.emissive_color * make_float3(sampled.x, sampled.y, sampled.z);
	}
	else
	{
		return material.emissive_color;
	}
}

extern "C" 
__global__ void RG_NAME(rg)()
{
	OptixTraversableHandle scene = params.traversable;
	float3 const  eye = params.cam_eye;
	uint2  const  pixel  = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
	uint2  const  screen = make_uint2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y);
	uint32 samples = params.sample_count;

	float3 radiance = make_float3(0.0f);
	float3 throughput = make_float3(1.0f);
	do
	{
		uint32 seed = tea<4>(pixel.y * screen.x + pixel.x, samples);
		float3 ray_origin = eye;
		float3 ray_direction = GetRayDirection(pixel, screen, seed);


		HitInfo hit_info{};
		hit_info.depth = 0;
		uint32 p0 = PackPointer0(&hit_info), p1 = PackPointer1(&hit_info);
		//							   params.max_depth
		for (uint32 depth = 0; depth < 1; ++depth)
		{
			Trace(scene, ray_origin, ray_direction, M_EPSILON, M_INF, p0, p1);
			if (!hit_info.hit)
			{
				float3 const& dir = ray_direction;
				float u = (1.f + atan2(dir.x, -dir.z) * M_INV_PI) * 0.5f;
				float v = 1.0f - acos(dir.y) * M_INV_PI;
				float3 env_map_color = make_float3(0.0f);
				if (params.sky)
				{
					float4 sampled = tex2D<float4>(params.sky, u, v);
					env_map_color = make_float3(sampled.x, sampled.y, sampled.z);
				}

				//#todo add MIS / power heuristic
				radiance += env_map_color * throughput;
				break;
			}

			if (depth == 0)
			{
				float3 emissive = GetEmissive(hit_info.material_idx, hit_info.uv);
				radiance += emissive * throughput;
			}

			//temporary
			MaterialGPU material = params.materials[hit_info.material_idx];
			if (material.diffuse_tex_id >= 0)
			{
				float4 sampled = tex2D<float4>(params.textures[material.diffuse_tex_id], hit_info.uv.x, hit_info.uv.y);
				radiance += material.base_color * make_float3(sampled.x, sampled.y, sampled.z);
			}
			else
			{
				radiance += material.base_color;
			}
			LightGPU light = params.lights[0];
			if (TraceOcclusion(params.traversable, hit_info.P + M_EPSILON * hit_info.N, -light.direction, 0.001f, 100000.0f))
			{
				radiance = make_float3(0.0f);
			}

			//#todo Next event estimation
			//radiance += SampleDirectLight(ray, hit_info) * throughput;

			//#todo Sample BSDF for color and outgoing direction
			//scatterSample.f = MaterialSample(state, -r.direction, state.ffnormal, scatterSample.L, scatterSample.pdf);
			//if (scatterSample.pdf > 0.0)
			//	throughput *= scatterSample.f / scatterSample.pdf;
			//else
			//	break;

			 // Move ray origin to hit point and set direction for next bounce
			//r.direction = scatterSample.L;
			//r.origin = state.fhp + r.direction * EPS;
			ray_origin = hit_info.P;
			ray_direction = make_float3(0.0f, 1.0f, 0.0f);

			//russian roulette
			if (depth >= RR_DEPTH)
			{
				float q = min(max(throughput.x, max(throughput.y, throughput.z)) + 0.001, 0.95);
				if (rnd(seed) > q) break;
				throughput /= q;
			}
		}
	} while (--samples);

	radiance = radiance / params.sample_count;
	params.image[pixel.x + pixel.y * screen.x] = MakeColor(radiance);
}

extern "C" 
__global__ void MISS_NAME(ms)()
{
	GetPayload<HitInfo>()->hit = false;
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


extern "C" 
__global__ void CH_NAME(ch)()
{
	uint32 instance_idx = optixGetInstanceIndex();
	uint32 primitive_idx = optixGetPrimitiveIndex();

	MeshGPU mesh = params.meshes[instance_idx];
	VertexData vertex = LoadVertexData(mesh, optixGetPrimitiveIndex(), optixGetTriangleBarycentrics());

	HitInfo* hit_info = GetPayload<HitInfo>();
	hit_info->hit = true;
	hit_info->P = vertex.P;
	hit_info->N = vertex.N;
	hit_info->uv = vertex.uv;
	hit_info->material_idx = mesh.material_idx;
}

