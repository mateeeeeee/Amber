#pragma once
#include "DeviceCommon.cuh"
#include "Device/DeviceHostCommon.h"
#include "Random.cuh"
#include "Color.cuh"
#include "ONB.cuh"
#include "Disney.cuh"

using namespace amber;

extern "C" 
{
	__constant__ LaunchParams params;
}

template<typename T>
__forceinline__ __device__ T Interpolate(T const& t0, T const& t1, T const& t2, float2 bary)
{
	return t0 * (1.0f - bary.x - bary.y) + bary.x * t1 + bary.y * t2;
}

__device__ __forceinline__ void UnpackMaterial(DisneyMaterial& mat_params, Uint32 id, float2 uv)
{
	MaterialGPU material = params.materials[id];
	if (material.diffuse_tex_id >= 0)
	{
		float4 sampled = tex2D<float4>(params.textures[material.diffuse_tex_id], uv.x, uv.y);
		mat_params.base_color = material.base_color * make_float3(sampled.x, sampled.y, sampled.z);
	}
	else
	{
		mat_params.base_color = material.base_color;
	}

	if (material.emissive_tex_id >= 0)
	{
		float4 sampled = tex2D<float4>(params.textures[material.emissive_tex_id], uv.x, uv.y);
		mat_params.emissive = material.emissive_color * make_float3(sampled.x, sampled.y, sampled.z);
	}
	else
	{
		mat_params.emissive = material.emissive_color;
	}

	if (material.metallic_roughness_tex_id >= 0)
	{
		float4 sampled = tex2D<float4>(params.textures[material.metallic_roughness_tex_id], uv.x, uv.y);
		mat_params.ao = sampled.x;
		mat_params.roughness = sampled.y * material.roughness;
		mat_params.metallic = sampled.z * material.metallic;
	}
	else
	{
		mat_params.ao = 1.0f;
		mat_params.roughness = material.roughness;
		mat_params.metallic = material.metallic;
	}

	if (material.normal_tex_id >= 0)
	{
		float4 sampled = tex2D<float4>(params.textures[material.normal_tex_id], uv.x, uv.y);
		mat_params.normal = make_float3(sampled.x, sampled.y, sampled.z);
		mat_params.normal = 2.0f * mat_params.normal - 1.0f;
	}
	else
	{
		mat_params.normal = make_float3(0.0f, 0.0f, 1.0f);
	}

	mat_params.specular_tint = material.specular_tint;
	mat_params.anisotropy = material.anisotropy;
	mat_params.sheen = material.sheen;
	mat_params.sheen_tint = material.sheen_tint;
	mat_params.clearcoat = material.clearcoat;
	mat_params.clearcoat_gloss = material.clearcoat_gloss;
	mat_params.ior = material.ior;
	mat_params.specular_transmission = material.specular_transmission;
}

__device__ __forceinline__ float3 ApplyNormalMap(const float3& normal_map, const float3& v_x, const float3& v_y, const float3& v_z)
{
	return normalize(normal_map.x * v_x + normal_map.y * v_y + normal_map.z * v_z);
}

__device__ __forceinline__ float3 GetRayDirection(uint2 pixel, uint2 screen, unsigned int seed)
{
	float3 const U = params.cam_u;
	float3 const V = params.cam_v;
	float3 const W = params.cam_w;

	float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));
	float2 d = (make_float2(pixel) + subpixel_jitter) / make_float2(screen);
	d.y = 1.0f - d.y;
	d = 2.0f * d - 1.0f;

	float tan_half_fovy = tan(params.cam_fovy * 0.5f);
	float aspect_ratio = params.cam_aspect_ratio;

	float3 ray_direction = normalize(d.x * aspect_ratio * tan_half_fovy * U + d.y * tan_half_fovy * V + W);
	return ray_direction;
}

__device__ __forceinline__ float3 SampleDirectLight(DisneyMaterial const& mat_params, float3 const& hit_point, float3 const& w_o, OrthonormalBasis const& ort, Uint32& seed)
{
	Uint32 light_index = rnd(seed) * params.light_count;
	LightGPU light = params.lights[light_index];

	float3 const& v_x = ort.tangent;
	float3 const& v_y = ort.binormal;
	float3 const& v_z = ort.normal;

	float3 radiance = make_float3(0.0f);
	if (light.type == LightType_Directional)
	{
		float3 light_dir = normalize(light.direction);
		if (!TraceOcclusion(params.traversable, hit_point + M_EPSILON * v_z, -light_dir, M_EPSILON, M_INF))
		{
			float3 bsdf = DisneyBrdf(mat_params, v_z, w_o, -light_dir, v_x, v_y);
			radiance = bsdf * light.color * abs(dot(-light_dir, v_z));
		}

		float3 w_i;
		float bsdf_pdf;
		float3 bsdf = SampleDisneyBrdf(mat_params, v_z, w_o, v_x, v_y, seed, w_i, bsdf_pdf);

		if (length(bsdf) > M_EPSILON && bsdf_pdf >= M_EPSILON)
		{
			float light_pdf = 1.0f; 
			float w = PowerHeuristic(1.f, bsdf_pdf, 1.f, light_pdf);

			if (!TraceOcclusion(params.traversable, hit_point + M_EPSILON * v_z, w_i, M_EPSILON, M_INF))
			{
				float3 bsdf = DisneyBrdf(mat_params, v_z, w_o, -light_dir, v_x, v_y);
				radiance += bsdf * light.color * abs(dot(w_i, v_z)) * w / bsdf_pdf;
			}
		}
	}
	else if (light.type == LightType_Point)
	{
		float3 light_pos = light.position;
		float3 light_dir = light_pos - hit_point; 
		float dist = length(light_dir);
		light_dir = light_dir / dist;

		if (!TraceOcclusion(params.traversable, hit_point + M_EPSILON * v_z, light_dir, M_EPSILON, dist - M_EPSILON))
		{
			float attenuation = 1.0f / (dist * dist);
			float3 bsdf = DisneyBrdf(mat_params, v_z, w_o, light_dir, v_x, v_y);
			radiance = bsdf * light.color * abs(dot(light_dir, v_z)) * attenuation;
		}

		float3 w_i;
		float bsdf_pdf;
		float3 bsdf = SampleDisneyBrdf(mat_params, v_z, w_o, v_x, v_y, seed, w_i, bsdf_pdf);
		if (length(bsdf) > M_EPSILON && bsdf_pdf >= M_EPSILON)
		{
			float light_pdf = (dist * dist) / (abs(dot(light_dir, v_z)) * 1.0f); // light.radius);
			float w = PowerHeuristic(1.f, bsdf_pdf, 1.f, light_pdf);

			if (!TraceOcclusion(params.traversable, hit_point + M_EPSILON * v_z, w_i, M_EPSILON, dist - M_EPSILON))
			{
				float attenuation = 1.0f / (dist * dist); 
				radiance += bsdf * light.color * abs(dot(w_i, v_z)) * attenuation * w / bsdf_pdf;
			}
		}
	}
	return radiance;
}

__device__ void WriteToDenoiserBuffers(Uint32 idx, float3 const& albedo, float3 const& normal)
{
	if (params.denoiser_albedo != NULL)
	{
		params.denoiser_albedo[idx] = albedo;
	}

	if (params.denoiser_normals != NULL)
	{
		float3 view_normal;
		view_normal.x = dot(normal,  params.cam_u);
		view_normal.y = dot(normal,  params.cam_v);
		view_normal.z = dot(normal, -params.cam_w);
		params.denoiser_normals[idx] = view_normal;
	}
}

__device__ void WriteToDebugBuffer(Uint32 idx, float3 const& albedo, float3 const& normal, float2 const& uv, Uint32 material_id)
{
	if (params.output_type == PathTracerOutputGPU_Albedo)
	{
		params.debug_buffer[idx] = albedo;
		return;
	}
	if (params.output_type == PathTracerOutputGPU_Normal)
	{
		params.debug_buffer[idx] = normal;
		return;
	}
	if (params.output_type == PathTracerOutputGPU_UV)
	{
		params.debug_buffer[idx] = make_float3(uv, 0.0f);
		return;
	}
	if(params.output_type == PathTracerOutputGPU_MaterialID)
	{
		float3 material_id_color = make_float3(
			(material_id * 37) % 255 / 255.0, 
			(material_id * 59) % 255 / 255.0,
			(material_id * 97) % 255 / 255.0);
		params.debug_buffer[idx] = material_id_color;
		return;
	}
}


extern "C" 
__global__ void RG_NAME(rg)()
{
	OptixTraversableHandle scene = params.traversable;
	float3 const  eye = params.cam_eye;
	uint2  const  pixel  = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
	uint2  const  screen = make_uint2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y);
	Uint32 samples = params.sample_count;
	Uint32 idx = pixel.x + pixel.y * screen.x;

	float3 radiance = make_float3(0.0f);
	float3 throughput = make_float3(1.0f);
	do
	{
		Uint32 seed = tea<4>(idx, samples + params.frame_index);
		float3 ray_origin = eye;
		float3 ray_direction = GetRayDirection(pixel, screen, seed);

		HitRecord hit_record{};
		hit_record.depth = 0;
		Uint32 p0 = PackPointer0(&hit_record), p1 = PackPointer1(&hit_record);

		for (Uint32 depth = 0; depth < params.max_depth; ++depth)
		{
			Trace(scene, ray_origin, ray_direction, M_EPSILON, M_INF, p0, p1);
			if (!hit_record.hit)
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

				radiance += env_map_color * throughput;

				if (depth == 0)
				{
					WriteToDenoiserBuffers(idx, make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f));
					WriteToDebugBuffer(idx, make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float2(0.0f, 0.0f), 0);
				}
				break;
			}

			DisneyMaterial material{};
			UnpackMaterial(material, hit_record.material_idx, hit_record.uv);

			float3 emissive = material.emissive;
			radiance += emissive * throughput;

			float3 w_o = -ray_direction;
			float3 v_x, v_y;
			float3 v_z = hit_record.N;
			if (material.specular_transmission == 0.0f && dot(w_o, v_z) < 0.0f)
			{
				v_z = -v_z;
			}

			OrthonormalBasis ort(v_z);
			v_x = ort.tangent;
			v_y = ort.binormal;

			//if (length(material.normal - make_float3(0.0f, 0.0f, 1.0f)) > 1e-4f)
			//{
			//	v_z = ApplyNormalMap(material.normal, v_x, v_y, v_z);
			//	ort = OrthonormalBasis(v_z);
			//	v_x = ort.tangent;
			//	v_y = ort.binormal;
			//}

			if (depth == 0)
			{
				WriteToDenoiserBuffers(idx, material.base_color, v_z);
				WriteToDebugBuffer(idx, material.base_color, v_z, hit_record.uv, hit_record.material_idx);
			}

			radiance += SampleDirectLight(material, hit_record.P, w_o, ort, seed) * throughput;

			float3 w_i;
			float pdf;
			float3 bsdf = SampleDisneyBrdf(material, v_z, w_o, v_x, v_y, seed, w_i, pdf);

			if (params.output_type == PathTracerOutputGPU_Custom && depth == 0)
			{
				bool entering = dot(w_o, v_z) > 0.f;
				float dot_wi_n = dot(w_i, v_z); // Sampled direction vs. normal
				bool is_reflected = SameHemisphere(w_o, w_i, v_z); // 1.0 if reflected, 0.0 if refracted
				params.debug_buffer[idx] = make_float3(dot_wi_n, is_reflected ? 1.0f : 0.0f, pdf);
				return;
			}

			if (pdf == 0.0f || length(bsdf) < M_EPSILON)
			{
				break;
			}
			throughput *= bsdf * abs(dot(w_i, v_z)) / pdf;
			
			ray_origin = hit_record.P + w_i * 1e-3;
			ray_direction = w_i;

			if (depth >= 2)
			{
				float q = min(max(throughput.x, max(throughput.y, throughput.z)) + 0.001f, 0.95f);
				if (rnd(seed) > q) break;
				throughput /= q;
			}
		}
	} while (--samples);

	radiance = radiance / params.sample_count;

	//temporary to reduce fireflies
	float lum = dot(radiance, make_float3(0.212671f, 0.715160f, 0.072169f));
	if (lum > 50.0f)
	{
		radiance *= 50.0f / lum;
	}

	float3 old_accum_color = params.accum_buffer[idx];
	if (params.frame_index > 0)
	{
		radiance += old_accum_color;
	}
	params.accum_buffer[idx] = radiance;
}

extern "C" 
__global__ void MISS_NAME(ms)()
{
	GetPayload<HitRecord>()->hit = false;
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
	Uint32 i0 = primitive_indices.x;
	Uint32 i1 = primitive_indices.y;
	Uint32 i2 = primitive_indices.z;

	float3* mesh_vertices = params.vertices + mesh.positions_offset;
	float3 pos0 = mesh_vertices[i0];
	float3 pos1 = mesh_vertices[i1];
	float3 pos2 = mesh_vertices[i2];
	vertex.P = Interpolate(pos0, pos1, pos2, barycentrics);

	//geometric normal
	//float3 edge1 = world_v1 - world_v0;
	//float3 edge2 = world_v2 - world_v0;
	//float3 geometric_normal = normalize(cross(edge1, edge2));

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
	vertex.uv.y = 1.0f - vertex.uv.y;
	return vertex;
}

extern "C" 
__global__ void AH_NAME(ah)()
{
	Uint32 instance_idx = optixGetInstanceId();
	Uint32 primitive_idx = optixGetPrimitiveIndex();

	MeshGPU mesh = params.meshes[instance_idx];
	VertexData vertex = LoadVertexData(mesh, optixGetPrimitiveIndex(), optixGetTriangleBarycentrics());
	MaterialGPU material = params.materials[mesh.material_idx];

	if (material.diffuse_tex_id >= 0)
	{
		float4 sampled = tex2D<float4>(params.textures[material.diffuse_tex_id], vertex.uv.x, vertex.uv.y);
		if(sampled.w < material.alpha_cutoff) optixIgnoreIntersection();
	}
}


__device__ float3 TransformVertex(float const matrix[12], float3 const& position)
{
	float3 transformed_position;
	transformed_position.x = matrix[0] * position.x + matrix[1] * position.y + matrix[2] * position.z + matrix[3];
	transformed_position.y = matrix[4] * position.x + matrix[5] * position.y + matrix[6] * position.z + matrix[7];
	transformed_position.z = matrix[8] * position.x + matrix[9] * position.y + matrix[10] * position.z + matrix[11];
	return transformed_position;
}

__device__ float3 TransformNormal(float const matrix[12], float3 const& normal)
{
	float3 transformed_normal;
	transformed_normal.x = matrix[0] * normal.x + matrix[1] * normal.y + matrix[2] * normal.z;
	transformed_normal.y = matrix[4] * normal.x + matrix[5] * normal.y + matrix[6] * normal.z;
	transformed_normal.z = matrix[8] * normal.x + matrix[9] * normal.y + matrix[10] * normal.z;
	return normalize(transformed_normal);
}

extern "C" 
__global__ void CH_NAME(ch)()
{
	MeshGPU mesh = params.meshes[optixGetInstanceId()];
	VertexData vertex = LoadVertexData(mesh, optixGetPrimitiveIndex(), optixGetTriangleBarycentrics());

	float object_to_world_transform[12];
	optixGetObjectToWorldTransformMatrix(object_to_world_transform);

	HitRecord* hit_record = GetPayload<HitRecord>();
	hit_record->hit = true;
	hit_record->P = TransformVertex(object_to_world_transform, vertex.P);
	hit_record->N = TransformNormal(object_to_world_transform, vertex.N);
	hit_record->uv = vertex.uv;
	hit_record->material_idx = mesh.material_idx;
}


extern "C"
__global__ void AH_NAME(ah_shadow)()
{
	Uint32 instance_idx = optixGetInstanceId();
	Uint32 primitive_idx = optixGetPrimitiveIndex();

	MeshGPU mesh = params.meshes[instance_idx];
	VertexData vertex = LoadVertexData(mesh, optixGetPrimitiveIndex(), optixGetTriangleBarycentrics());
	MaterialGPU material = params.materials[mesh.material_idx];

	if (material.diffuse_tex_id >= 0)
	{
		float4 sampled = tex2D<float4>(params.textures[material.diffuse_tex_id], vertex.uv.x, vertex.uv.y);
		if (sampled.w < material.alpha_cutoff) optixIgnoreIntersection();
	}
	if (material.specular_transmission > 0)
	{
		optixIgnoreIntersection();
	}
}
