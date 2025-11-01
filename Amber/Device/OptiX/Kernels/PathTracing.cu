#pragma once
#include "DeviceCommon.cuh"
#include "Device/OptiX/DeviceHostCommon.h"
#include "Disney.cuh"
#include "ONB.cuh"

using namespace amber;

extern "C" __constant__ LaunchParams params;

__device__ __forceinline__ Float3 ApplyNormalMapping(Float3 tangent_space_normal, Float3 N, Float3 T, Float3 B)
{
	return normalize(tangent_space_normal.x * T + tangent_space_normal.y * B + tangent_space_normal.z * N);
}
__device__ __forceinline__ void EvaluateMaterial(MaterialGPU const& material, EvaluatedMaterial& evaluated_material, Float2 uv)
{
	if (material.diffuse_tex_id >= 0)
	{
		Float4 sampled = tex2D<Float4>(params.textures[material.diffuse_tex_id], uv.x, uv.y);
		evaluated_material.base_color = material.base_color * MakeFloat3(sampled.x, sampled.y, sampled.z);
	}
	else
	{
		evaluated_material.base_color = material.base_color;
	}

	if (material.emissive_tex_id >= 0)
	{
		Float4 sampled = tex2D<Float4>(params.textures[material.emissive_tex_id], uv.x, uv.y);
		evaluated_material.emissive = material.emissive_color * MakeFloat3(sampled.x, sampled.y, sampled.z);
	}
	else
	{
		evaluated_material.emissive = material.emissive_color;
	}

	if (material.metallic_roughness_tex_id >= 0)
	{
		Float4 sampled = tex2D<Float4>(params.textures[material.metallic_roughness_tex_id], uv.x, uv.y);
		evaluated_material.ao = sampled.x;
		evaluated_material.roughness = sampled.y * material.roughness;
		evaluated_material.metallic = sampled.z * material.metallic;
	}
	else
	{
		evaluated_material.ao = 1.0f;
		evaluated_material.roughness = material.roughness;
		evaluated_material.metallic = material.metallic;
	}

	if (material.normal_tex_id >= 0)
	{
		Float4 sampled = tex2D<Float4>(params.textures[material.normal_tex_id], uv.x, uv.y);
		evaluated_material.tangent_space_normal = MakeFloat3(sampled.x, sampled.y, sampled.z);
		evaluated_material.tangent_space_normal = 2.0f * evaluated_material.tangent_space_normal - 1.0f;
	}
	else
	{
		evaluated_material.tangent_space_normal = MakeFloat3(0.0f, 0.0f, 1.0f);
	}

	evaluated_material.specular_tint = material.specular_tint;
	evaluated_material.anisotropy = material.anisotropy;
	evaluated_material.sheen = material.sheen;
	evaluated_material.sheen_tint = material.sheen_tint;
	evaluated_material.clearcoat = material.clearcoat;
	evaluated_material.clearcoat_gloss = material.clearcoat_gloss;
	evaluated_material.ior = material.ior;
	evaluated_material.specular_transmission = material.specular_transmission;
}
__device__ __forceinline__ Float3 GetRayDirection(Uint2 pixel, Uint2 screen, PRNG& prng)
{
	Float3 const U = params.cam_u;
	Float3 const V = params.cam_v;
	Float3 const W = params.cam_w;

	Float2 subpixelJitter = prng.RandomFloat2();
	Float2 d = (MakeFloat2(pixel) + subpixelJitter) / MakeFloat2(screen);
	d.y = 1.0f - d.y;
	d = 2.0f * d - 1.0f;
	Float tanHalfFovy = tan(params.cam_fovy * 0.5f);
	Float aspectRatio = params.cam_aspect_ratio;
	Float3 ray_direction = normalize(d.x * aspectRatio * tanHalfFovy * U + d.y * tanHalfFovy * V + W);
	return ray_direction;
}

__device__ __forceinline__ ColorRGB32F SampleDirectLight(EvaluatedMaterial const& evaluated_material, Float3 const& hit_point, Float3 const& wo, 
	Float3 const& T, Float3 const& B, Float3 const& N, PRNG& prng)
{
	Uint32 light_index = prng.RandomFloat() * params.light_count;
	LightGPU light = params.lights[light_index];

	ColorRGB32F radiance(0.0f);
	if (light.type == LightGPUType_Directional)
	{
		Float3 light_direction = normalize(light.direction);
		if (!TraceOcclusion(params.traversable, hit_point + M_EPSILON * N, -light_direction, M_EPSILON, M_INF))
		{
			ColorRGB32F bsdf = DisneyBrdf(evaluated_material, N, wo, -light_direction, T, B);
			radiance = bsdf * light.color * abs(dot(-light_direction, N));
		}

		Float3 wi;
		Float bsdf_pdf;
		ColorRGB32F bsdf = SampleDisneyBrdf(evaluated_material, N, wo, T, B, prng, wi, bsdf_pdf);
		if (bsdf.Length() > M_EPSILON && bsdf_pdf >= M_EPSILON)
		{
			Float light_pdf = 1.0f; 
			Float w = PowerHeuristic(1.f, bsdf_pdf, 1.f, light_pdf);

			if (!TraceOcclusion(params.traversable, hit_point + M_EPSILON * N, wi, M_EPSILON, M_INF))
			{
				ColorRGB32F bsdf = DisneyBrdf(evaluated_material, N, wo, -light_direction, T, B);
				radiance += bsdf * light.color * abs(dot(wi, N)) * w / bsdf_pdf;
			}
		}
	}
	else if (light.type == LightGPUType_Point)
	{
		Float3 light_pos = light.position;
		Float3 light_dir = light_pos - hit_point; 
		Float dist = length(light_dir);
		light_dir = light_dir / dist;

		if (!TraceOcclusion(params.traversable, hit_point + M_EPSILON * N, light_dir, M_EPSILON, dist - M_EPSILON))
		{
			Float attenuation = 1.0f / (dist * dist);
			ColorRGB32F bsdf = DisneyBrdf(evaluated_material, N, wo, light_dir, T, B);
			radiance = bsdf * light.color * abs(dot(light_dir, N)) * attenuation;
		}

		Float3 w_i;
		Float bsdf_pdf;
		ColorRGB32F bsdf = SampleDisneyBrdf(evaluated_material, N, wo, T, B, prng, w_i, bsdf_pdf);
		if (bsdf.Length() > M_EPSILON && bsdf_pdf >= M_EPSILON)
		{
			Float light_pdf = (dist * dist) / (abs(dot(light_dir, N)) * 1.0f); // light.radius);
			Float w = PowerHeuristic(1.f, bsdf_pdf, 1.f, light_pdf);

			if (!TraceOcclusion(params.traversable, hit_point + M_EPSILON * N, w_i, M_EPSILON, dist - M_EPSILON))
			{
				Float attenuation = 1.0f / (dist * dist); 
				radiance += bsdf * light.color * abs(dot(w_i, N)) * attenuation * w / bsdf_pdf;
			}
		}
	}
	return radiance;
}

__device__ __forceinline__ void WriteToDenoiserBuffers(Uint32 idx, Float3 const& albedo, Float3 const& normal)
{
	if (params.denoiser_albedo != NULL)
	{
		params.denoiser_albedo[idx] = albedo;
	}

	if (params.denoiser_normals != NULL)
	{
		Float3 view_normal;
		view_normal.x = dot(normal,  params.cam_u);
		view_normal.y = dot(normal,  params.cam_v);
		view_normal.z = dot(normal, -params.cam_w);
		params.denoiser_normals[idx] = view_normal;
	}
}
__device__ __forceinline__ void WriteToDebugBuffer(Uint32 idx, Float3 const& albedo, Float3 const& normal, Float2 const& uv, Uint32 material_id)
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
		params.debug_buffer[idx] = MakeFloat3(uv, 0.0f);
		return;
	}
	if(params.output_type == PathTracerOutputGPU_MaterialID)
	{
		Float3 material_id_color = MakeFloat3(
			(material_id * 37) % 255 / 255.0, 
			(material_id * 59) % 255 / 255.0,
			(material_id * 97) % 255 / 255.0);
		params.debug_buffer[idx] = material_id_color;
		return;
	}
}


extern "C" __global__ void RG_NAME(rg)()
{
	OptixTraversableHandle scene = params.traversable;
	Float3 const  eye = params.cam_eye;
	Uint2  const  pixel  = MakeUint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
	Uint2  const  screen = MakeUint2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y);
	Uint32 samples = params.sample_count;
	Uint32 idx = pixel.x + pixel.y * screen.x;

	ColorRGB32F radiance(0.0f);
	ColorRGB32F throughput(1.0f);
	do
	{
		PRNG prng = PRNG::Create(idx, samples + params.frame_index);
		Float3 ray_origin = eye;
		Float3 ray_direction = GetRayDirection(pixel, screen, prng);

		HitRecord hit_record{};
		Uint32 p0 = PackPointer0(&hit_record), p1 = PackPointer1(&hit_record);

		for (Uint32 depth = 0; depth < params.max_depth; ++depth)
		{
			Trace(scene, ray_origin, ray_direction, M_EPSILON, M_INF, p0, p1);
			if (!hit_record.hit)
			{
				Float3 const& dir = ray_direction;
				Float u = (1.f + atan2(dir.x, -dir.z) * M_INV_PI) * 0.5f;
				Float v = 1.0f - acos(dir.y) * M_INV_PI;
				Float3 env_map_color = MakeFloat3(0.0f);
				if (params.sky)
				{
					Float4 sampled = tex2D<Float4>(params.sky, u, v);
					env_map_color = MakeFloat3(sampled.x, sampled.y, sampled.z);
				}
				radiance += env_map_color * throughput;

				if (depth == 0)
				{
					WriteToDenoiserBuffers(idx, MakeFloat3(0.0f, 0.0f, 0.0f), MakeFloat3(0.0f, 0.0f, 0.0f));
					WriteToDebugBuffer(idx, MakeFloat3(0.0f, 0.0f, 0.0f), MakeFloat3(0.0f, 0.0f, 0.0f), MakeFloat2(0.0f, 0.0f), 0);
				}
				break;
			}

			MeshGPU mesh = params.meshes[hit_record.instance_idx];
			MaterialGPU material_gpu = params.materials[mesh.material_idx];

			EvaluatedMaterial material{};
			EvaluateMaterial(material_gpu, material, hit_record.uv);

			ColorRGB32F emissive = material.emissive;
			radiance += emissive * throughput;

			Float3 w_o = -ray_direction;
			Float3 T, B;
			Float3 Ns = hit_record.Ns;
			Float3 Ng = hit_record.Ng;
			if (material.specular_transmission == 0.0f && dot(w_o, Ns) < 0.0f)
			{
				Ns = -Ns;
				Ng = -Ng;
			}
			BuildONB(Ns, T, B);
			Ns = ApplyNormalMapping(material.tangent_space_normal, Ns, T, B);

			if (depth == 0)
			{
				WriteToDenoiserBuffers(idx, (Float3)material.base_color, Ns);
				WriteToDebugBuffer(idx, (Float3)material.base_color, Ng, hit_record.uv, mesh.material_idx);
			}

			radiance += SampleDirectLight(material, hit_record.P, w_o, T, B, Ns, prng) * throughput;

			Float3 w_i;
			Float pdf;
			ColorRGB32F bsdf = SampleDisneyBrdf(material, Ns, w_o, T, B, prng, w_i, pdf);
			if (params.output_type == PathTracerOutputGPU_Custom && depth == 0)
			{
				Bool entering = dot(w_o, Ns) > 0.0f;
				Float dot_wi_n = dot(w_i, Ns); 
				Bool is_reflected = SameHemisphere(w_o, w_i, Ns); 
				params.debug_buffer[idx] = MakeFloat3(is_reflected, is_reflected, is_reflected);
				return;
			}

			if (pdf == 0.0f || bsdf.Length() < M_EPSILON)
			{
				break;
			}
			throughput *= bsdf * abs(dot(w_i, Ns)) / pdf;
			
			ray_origin = hit_record.P + w_i * 1e-3;
			ray_direction = w_i;

			if (depth >= 2)
			{
				Float q = min(max(throughput.r, max(throughput.g, throughput.b)) + 0.001f, 0.95f);
				if (prng.RandomFloat() > q) break;
				throughput /= q;
			}
		}
	} while (--samples);

	radiance = radiance / params.sample_count;

	Float luminance = radiance.Luminance(); 
	if (luminance > 50.0f)
	{
		radiance *= 50.0f / luminance;
	}
	Float3 old_accum_color = params.accum_buffer[idx];
	if (params.frame_index > 0)
	{
		radiance += old_accum_color;
	}
	params.accum_buffer[idx] = static_cast<Float3>(radiance);
}

extern "C" __global__ void MISS_NAME(ms)()
{
	GetPayload<HitRecord>()->hit = false;
}

struct VertexData
{
	Float3 P;
	Float3 Ng;
	Float3 Ns;
	Float2 uv;
};
__device__ VertexData LoadVertexData(MeshGPU const& mesh, Uint32 primitive_idx, Float2 barycentrics)
{
	VertexData vertex{};
	Uint3* mesh_indices = params.indices + mesh.indices_offset;

	Uint3 primitive_indices = mesh_indices[primitive_idx];
	Uint32 i0 = primitive_indices.x;
	Uint32 i1 = primitive_indices.y;
	Uint32 i2 = primitive_indices.z;

	Float3* mesh_vertices = params.vertices + mesh.positions_offset;
	Float3 pos0 = mesh_vertices[i0];
	Float3 pos1 = mesh_vertices[i1];
	Float3 pos2 = mesh_vertices[i2];
	vertex.P = Interpolate(pos0, pos1, pos2, barycentrics);

	Float3 edge1 = pos1 - pos0;
	Float3 edge2 = pos2 - pos0;
	vertex.Ng = normalize(cross(edge1, edge2));

	Float3* mesh_normals = params.normals + mesh.normals_offset;
	Float3 nor0 = mesh_normals[i0];
	Float3 nor1 = mesh_normals[i1];
	Float3 nor2 = mesh_normals[i2];
	vertex.Ns = Interpolate(nor0, nor1, nor2, barycentrics);
	
	Float2* mesh_uvs = params.uvs + mesh.uvs_offset;
	Float2 uv0 = mesh_uvs[i0];
	Float2 uv1 = mesh_uvs[i1];
	Float2 uv2 = mesh_uvs[i2];
	vertex.uv = Interpolate(uv0, uv1, uv2, barycentrics);
	vertex.uv.y = 1.0f - vertex.uv.y;
	return vertex;
}

extern "C"  __global__ void AH_NAME(ah)()
{
	Uint32 instance_idx = optixGetInstanceId();
	Uint32 primitive_idx = optixGetPrimitiveIndex();

	MeshGPU mesh = params.meshes[instance_idx];
	MaterialGPU material = params.materials[mesh.material_idx];
	if (material.diffuse_tex_id >= 0)
	{
		VertexData vertex = LoadVertexData(mesh, primitive_idx, optixGetTriangleBarycentrics());
		Float4 sampled = tex2D<Float4>(params.textures[material.diffuse_tex_id], vertex.uv.x, vertex.uv.y);
		if (sampled.w < material.alpha_cutoff)
		{
			optixIgnoreIntersection();
		}
	}
}

__device__ Float3 TransformVertex(Float const matrix[12], Float3 const& position)
{
	Float3 transformed_position;
	transformed_position.x = matrix[0] * position.x + matrix[1] * position.y + matrix[2] * position.z + matrix[3];
	transformed_position.y = matrix[4] * position.x + matrix[5] * position.y + matrix[6] * position.z + matrix[7];
	transformed_position.z = matrix[8] * position.x + matrix[9] * position.y + matrix[10] * position.z + matrix[11];
	return transformed_position;
}
__device__ Float3 TransformNormal(Float const matrix[12], Float3 const& normal)
{
	Float3 transformed_normal;
	transformed_normal.x = matrix[0] * normal.x + matrix[1] * normal.y + matrix[2] * normal.z;
	transformed_normal.y = matrix[4] * normal.x + matrix[5] * normal.y + matrix[6] * normal.z;
	transformed_normal.z = matrix[8] * normal.x + matrix[9] * normal.y + matrix[10] * normal.z;
	return normalize(transformed_normal);
}

extern "C" __global__ void CH_NAME(ch)()
{
	Uint32 instance_idx = optixGetInstanceId();
	Uint32 primitive_idx = optixGetPrimitiveIndex();
	Float2 barycentrics = optixGetTriangleBarycentrics();

	MeshGPU mesh = params.meshes[instance_idx];
	MaterialGPU material = params.materials[mesh.material_idx];
	VertexData vertex = LoadVertexData(mesh, primitive_idx, barycentrics);

	Float object_to_world_transform[12];
	optixGetObjectToWorldTransformMatrix(object_to_world_transform);

	HitRecord* hit_record = GetPayload<HitRecord>();
	hit_record->P = TransformVertex(object_to_world_transform, vertex.P);
	hit_record->Ng = TransformNormal(object_to_world_transform, vertex.Ng);
	hit_record->uv = vertex.uv;
	hit_record->barycentrics = barycentrics;
	hit_record->primitive_idx = primitive_idx;
	hit_record->instance_idx = instance_idx;
	hit_record->hit = true;
	hit_record->t = optixGetRayTmax();
	hit_record->Ns = TransformNormal(object_to_world_transform, vertex.Ns);
}

extern "C" __global__ void AH_NAME(ah_shadow)()
{
	Uint32 instance_idx = optixGetInstanceId();
	Uint32 primitive_idx = optixGetPrimitiveIndex();
	Float2 barycentrics = optixGetTriangleBarycentrics();

	MeshGPU mesh = params.meshes[instance_idx];
	MaterialGPU material = params.materials[mesh.material_idx];
	if (material.diffuse_tex_id >= 0)
	{
		VertexData vertex = LoadVertexData(mesh, primitive_idx, barycentrics);
		Float4 sampled = tex2D<Float4>(params.textures[material.diffuse_tex_id], vertex.uv.x, vertex.uv.y);
		if (sampled.w < material.alpha_cutoff)
		{
			optixIgnoreIntersection();
		}
	}
	if (material.specular_transmission > 0)
	{
		optixIgnoreIntersection();
	}
}


/*
__device__ __forceinline__ Float3 ApplyNormalMapping(HitRecord const& hit_record)
{
	MeshGPU const& mesh = params.meshes[hit_record.instance_idx];
	Uint32 primitive_idx = hit_record.primitive_idx;
	Float2 barycentrics = hit_record.barycentrics;
	Float2 uv = hit_record.uv;
	Float3 N = hit_record.Ns;
	MaterialGPU material_gpu = params.materials[mesh.material_idx];
	if (material_gpu.normal_tex_id < 0)
	{
		return N;
	}
	Float4 sampled = tex2D<Float4>(params.textures[material_gpu.normal_tex_id], uv.x, uv.y);
	Float3 normal_tangent_space = MakeFloat3(sampled.x, sampled.y, sampled.z);
	normal_tangent_space = 2.0f * normal_tangent_space - 1.0f;

	MaterialGPU material = params.materials[mesh.material_idx];
	Uint3* mesh_indices = params.indices + mesh.indices_offset;

	Uint3 primitive_indices = mesh_indices[primitive_idx];
	Uint32 i0 = primitive_indices.x;
	Uint32 i1 = primitive_indices.y;
	Uint32 i2 = primitive_indices.z;

	Float2* mesh_uvs = params.uvs + mesh.uvs_offset;
	Float2 uv0 = mesh_uvs[i0];
	Float2 uv1 = mesh_uvs[i1];
	Float2 uv2 = mesh_uvs[i2];

	Float2 deltaUV_10 = uv1 - uv0;
	Float2 deltaUV_20 = uv2 - uv0;

	Float3* mesh_vertices = params.vertices + mesh.positions_offset;
	Float3 P0 = mesh_vertices[i0];
	Float3 P1 = mesh_vertices[i1];
	Float3 P2 = mesh_vertices[i2];

	Float3 edge_P0P1 = P1 - P0;
	Float3 edge_P0P2 = P2 - P0;

	float det_inverse = 1.0f / (deltaUV_10.x * deltaUV_20.y - deltaUV_10.y * deltaUV_20.x);
	float3 T = (edge_P0P1 * deltaUV_20.y - edge_P0P2 * deltaUV_10.y) * det_inverse;
	float3 B = (edge_P0P2 * deltaUV_10.x - edge_P0P1 * deltaUV_20.x) * det_inverse;
	return LocalToWorldFrame(T, B, N, normal_tangent_space);
}*/
