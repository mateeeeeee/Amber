#pragma once
#include "DeviceCommon.cuh"
#include "Device/DeviceHostCommon.h"
#include "Disney.cuh"

using namespace amber;

extern "C" 
{
	__constant__ LaunchParams params;
}

__device__ __forceinline__ void EvaluateMaterial(EvaluatedMaterial& evaluatedMaterial, Uint32 id, Float2 uv)
{
	MaterialGPU material = params.materials[id];
	if (material.diffuse_tex_id >= 0)
	{
		Float4 sampled = tex2D<Float4>(params.textures[material.diffuse_tex_id], uv.x, uv.y);
		evaluatedMaterial.base_color = material.base_color * MakeFloat3(sampled.x, sampled.y, sampled.z);
	}
	else
	{
		evaluatedMaterial.base_color = material.base_color;
	}

	if (material.emissive_tex_id >= 0)
	{
		Float4 sampled = tex2D<Float4>(params.textures[material.emissive_tex_id], uv.x, uv.y);
		evaluatedMaterial.emissive = material.emissive_color * MakeFloat3(sampled.x, sampled.y, sampled.z);
	}
	else
	{
		evaluatedMaterial.emissive = material.emissive_color;
	}

	if (material.metallic_roughness_tex_id >= 0)
	{
		Float4 sampled = tex2D<Float4>(params.textures[material.metallic_roughness_tex_id], uv.x, uv.y);
		evaluatedMaterial.ao = sampled.x;
		evaluatedMaterial.roughness = sampled.y * material.roughness;
		evaluatedMaterial.metallic = sampled.z * material.metallic;
	}
	else
	{
		evaluatedMaterial.ao = 1.0f;
		evaluatedMaterial.roughness = material.roughness;
		evaluatedMaterial.metallic = material.metallic;
	}

	if (material.normal_tex_id >= 0)
	{
		Float4 sampled = tex2D<Float4>(params.textures[material.normal_tex_id], uv.x, uv.y);
		evaluatedMaterial.normal = MakeFloat3(sampled.x, sampled.y, sampled.z);
		evaluatedMaterial.normal = 2.0f * evaluatedMaterial.normal - 1.0f;
	}
	else
	{
		evaluatedMaterial.normal = MakeFloat3(0.0f, 0.0f, 1.0f);
	}

	evaluatedMaterial.specular_tint = material.specular_tint;
	evaluatedMaterial.anisotropy = material.anisotropy;
	evaluatedMaterial.sheen = material.sheen;
	evaluatedMaterial.sheen_tint = material.sheen_tint;
	evaluatedMaterial.clearcoat = material.clearcoat;
	evaluatedMaterial.clearcoat_gloss = material.clearcoat_gloss;
	evaluatedMaterial.ior = material.ior;
	evaluatedMaterial.specular_transmission = material.specular_transmission;
}
__device__ __forceinline__ Float3 ApplyNormalMap(Float3 const& normalMap, Float3 const& T, Float3 const& B, Float3 const& N)
{
	return normalize(normalMap.x * T + normalMap.y * B + normalMap.z * N);
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
__device__ __forceinline__ Float3 SampleDirectLight(EvaluatedMaterial const& evaluatedMaterial, Float3 const& hitPoint, Float3 const& wo, OrthonormalBasis const& ort, PRNG& prng)
{
	Uint32 lightIndex = prng.RandomFloat() * params.light_count;
	LightGPU light = params.lights[lightIndex];

	Float3 const& v_x = ort.tangent;
	Float3 const& v_y = ort.binormal;
	Float3 const& v_z = ort.normal;
	Float3 radiance = MakeFloat3(0.0f);
	if (light.type == LightGPUType_Directional)
	{
		Float3 lightDirection = normalize(light.direction);
		if (!TraceOcclusion(params.traversable, hitPoint + M_EPSILON * v_z, -lightDirection, M_EPSILON, M_INF))
		{
			Float3 bsdf = DisneyBrdf(evaluatedMaterial, v_z, wo, -lightDirection, v_x, v_y);
			radiance = bsdf * light.color * abs(dot(-lightDirection, v_z));
		}

		Float3 wi;
		Float bsdfPdf;
		Float3 bsdf = SampleDisneyBrdf(evaluatedMaterial, v_z, wo, v_x, v_y, prng, wi, bsdfPdf);

		if (length(bsdf) > M_EPSILON && bsdfPdf >= M_EPSILON)
		{
			Float light_pdf = 1.0f; 
			Float w = PowerHeuristic(1.f, bsdfPdf, 1.f, light_pdf);

			if (!TraceOcclusion(params.traversable, hitPoint + M_EPSILON * v_z, wi, M_EPSILON, M_INF))
			{
				Float3 bsdf = DisneyBrdf(evaluatedMaterial, v_z, wo, -lightDirection, v_x, v_y);
				radiance += bsdf * light.color * abs(dot(wi, v_z)) * w / bsdfPdf;
			}
		}
	}
	else if (light.type == LightGPUType_Point)
	{
		Float3 light_pos = light.position;
		Float3 light_dir = light_pos - hitPoint; 
		Float dist = length(light_dir);
		light_dir = light_dir / dist;

		if (!TraceOcclusion(params.traversable, hitPoint + M_EPSILON * v_z, light_dir, M_EPSILON, dist - M_EPSILON))
		{
			Float attenuation = 1.0f / (dist * dist);
			Float3 bsdf = DisneyBrdf(evaluatedMaterial, v_z, wo, light_dir, v_x, v_y);
			radiance = bsdf * light.color * abs(dot(light_dir, v_z)) * attenuation;
		}

		Float3 w_i;
		Float bsdf_pdf;
		Float3 bsdf = SampleDisneyBrdf(evaluatedMaterial, v_z, wo, v_x, v_y, prng, w_i, bsdf_pdf);
		if (length(bsdf) > M_EPSILON && bsdf_pdf >= M_EPSILON)
		{
			Float light_pdf = (dist * dist) / (abs(dot(light_dir, v_z)) * 1.0f); // light.radius);
			Float w = PowerHeuristic(1.f, bsdf_pdf, 1.f, light_pdf);

			if (!TraceOcclusion(params.traversable, hitPoint + M_EPSILON * v_z, w_i, M_EPSILON, dist - M_EPSILON))
			{
				Float attenuation = 1.0f / (dist * dist); 
				radiance += bsdf * light.color * abs(dot(w_i, v_z)) * attenuation * w / bsdf_pdf;
			}
		}
	}
	return radiance;
}

__device__ void WriteToDenoiserBuffers(Uint32 idx, Float3 const& albedo, Float3 const& normal)
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
__device__ void WriteToDebugBuffer(Uint32 idx, Float3 const& albedo, Float3 const& normal, Float2 const& uv, Uint32 material_id)
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

	Float3 radiance = MakeFloat3(0.0f);
	Float3 throughput = MakeFloat3(1.0f);
	do
	{
		PRNG prng = PRNG::Create(idx, samples + params.frame_index);
		Float3 ray_origin = eye;
		Float3 ray_direction = GetRayDirection(pixel, screen, prng);

		HitRecord hit_record{};
		hit_record.depth = 0;
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

			EvaluatedMaterial material{};
			EvaluateMaterial(material, hit_record.material_idx, hit_record.uv);

			Float3 emissive = material.emissive;
			radiance += emissive * throughput;

			Float3 w_o = -ray_direction;
			Float3 v_x, v_y;
			Float3 v_z = hit_record.N;
			if (material.specular_transmission == 0.0f && dot(w_o, v_z) < 0.0f)
			{
				v_z = -v_z;
			}

			OrthonormalBasis ort(v_z);
			v_x = ort.tangent;
			v_y = ort.binormal;

			//if (length(material.normal - MakeFloat3(0.0f, 0.0f, 1.0f)) > 1e-4f)
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

			radiance += SampleDirectLight(material, hit_record.P, w_o, ort, prng) * throughput;

			Float3 w_i;
			Float pdf;
			Float3 bsdf = SampleDisneyBrdf(material, v_z, w_o, v_x, v_y, prng, w_i, pdf);
			if (params.output_type == PathTracerOutputGPU_Custom && depth == 0)
			{
				Bool entering = dot(w_o, v_z) > 0.f;
				Float dot_wi_n = dot(w_i, v_z); // Sampled direction vs. normal
				Bool is_reflected = SameHemisphere(w_o, w_i, v_z); // 1.0 if reflected, 0.0 if refracted
				params.debug_buffer[idx] = MakeFloat3(dot_wi_n, is_reflected ? 1.0f : 0.0f, pdf);
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
				Float q = min(max(throughput.x, max(throughput.y, throughput.z)) + 0.001f, 0.95f);
				if (prng.RandomFloat() > q) break;
				throughput /= q;
			}
		}
	} while (--samples);

	radiance = radiance / params.sample_count;

	//temporary to reduce fireflies
	Float lum = dot(radiance, MakeFloat3(0.212671f, 0.715160f, 0.072169f));
	if (lum > 50.0f)
	{
		radiance *= 50.0f / lum;
	}

	Float3 old_accum_color = params.accum_buffer[idx];
	if (params.frame_index > 0)
	{
		radiance += old_accum_color;
	}
	params.accum_buffer[idx] = radiance;
}

extern "C" __global__ void MISS_NAME(ms)()
{
	GetPayload<HitRecord>()->hit = false;
}

struct VertexData
{
	Float3 P;
	Float3 N;
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

	//geometric normal
	//Float3 edge1 = world_v1 - world_v0;
	//Float3 edge2 = world_v2 - world_v0;
	//Float3 geometric_normal = normalize(cross(edge1, edge2));

	Float3* mesh_normals = params.normals + mesh.normals_offset;
	Float3 nor0 = mesh_normals[i0];
	Float3 nor1 = mesh_normals[i1];
	Float3 nor2 = mesh_normals[i2];
	vertex.N = Interpolate(nor0, nor1, nor2, barycentrics);
	
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
	VertexData vertex = LoadVertexData(mesh, optixGetPrimitiveIndex(), optixGetTriangleBarycentrics());
	MaterialGPU material = params.materials[mesh.material_idx];

	if (material.diffuse_tex_id >= 0)
	{
		Float4 sampled = tex2D<Float4>(params.textures[material.diffuse_tex_id], vertex.uv.x, vertex.uv.y);
		if(sampled.w < material.alpha_cutoff) optixIgnoreIntersection();
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
	MeshGPU mesh = params.meshes[optixGetInstanceId()];
	VertexData vertex = LoadVertexData(mesh, optixGetPrimitiveIndex(), optixGetTriangleBarycentrics());

	Float object_to_world_transform[12];
	optixGetObjectToWorldTransformMatrix(object_to_world_transform);

	HitRecord* hit_record = GetPayload<HitRecord>();
	hit_record->hit = true;
	hit_record->P = TransformVertex(object_to_world_transform, vertex.P);
	hit_record->N = TransformNormal(object_to_world_transform, vertex.N);
	hit_record->uv = vertex.uv;
	hit_record->material_idx = mesh.material_idx;
}

extern "C" __global__ void AH_NAME(ah_shadow)()
{
	Uint32 instance_idx = optixGetInstanceId();
	Uint32 primitive_idx = optixGetPrimitiveIndex();

	MeshGPU mesh = params.meshes[instance_idx];
	VertexData vertex = LoadVertexData(mesh, optixGetPrimitiveIndex(), optixGetTriangleBarycentrics());
	MaterialGPU material = params.materials[mesh.material_idx];

	if (material.diffuse_tex_id >= 0)
	{
		Float4 sampled = tex2D<Float4>(params.textures[material.diffuse_tex_id], vertex.uv.x, vertex.uv.y);
		if (sampled.w < material.alpha_cutoff) optixIgnoreIntersection();
	}
	if (material.specular_transmission > 0)
	{
		optixIgnoreIntersection();
	}
}
