#pragma once
#include "DeviceCommon.cuh"
#include "Device/OptiX/DeviceHostCommon.h"

using namespace amber;

extern "C" __constant__ LaunchParams params;

__device__ constexpr Float INV_PI = 0.31830988618379067154f;
__device__ constexpr Float EPSILON = 1e-5f;

template<Uint32 N>
__device__ __forceinline__ Uint32 Tea(Uint32 val0, Uint32 val1)
{
	Uint32 v0 = val0;
	Uint32 v1 = val1;
	Uint32 s0 = 0;
	for (Uint32 n = 0; n < N; n++)
	{
		s0 += 0x9e3779b9;
		v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
		v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
	}
	return v0;
}

__device__ __forceinline__ Uint32 LCG(Uint32& prev)
{
	constexpr Uint32 LCG_A = 1664525u;
	constexpr Uint32 LCG_C = 1013904223u;
	prev = (LCG_A * prev + LCG_C);
	return prev & 0x00FFFFFF;
}

struct PRNG
{
	Uint32 seed;

	__device__ static PRNG Create(Uint32 pixel_idx, Uint32 frame)
	{
		PRNG prng;
		prng.seed = Tea<4>(pixel_idx, frame);
		return prng;
	}

	__device__ Float RandomFloat()
	{
		return (Float)LCG(seed) / (Float)0x01000000;
	}

	__device__ Float2 RandomFloat2()
	{
		return MakeFloat2(RandomFloat(), RandomFloat());
	}
};

struct BxDFSample
{
	Float3 L;
	Float3 BxDF;
	Float PDF;
};

struct BxDFEval
{
	Float3 BxDF;
	Float PDF;
};

enum BSDFComponent
{
	Metallic,
	Diffuse,
	SpecularDielectric,
	GlassReflect,
	GlassRefract
};

struct EvaluatedMaterial
{
	Float3 base_color;
	Float3 emissive;
	Float3 specular_color;
	Float metallic;
	Float roughness;
	Float transmission;
	Float ior;
	Float anisotropy;
	Float Ax;
	Float Ay;
	Float Eta;
};

__device__ __forceinline__ Float Pow2(Float x)
{
	return x * x;
}

__device__ __forceinline__ Float3 CosSampleHemisphere(Float2 u)
{
	Float phi = 2.0f * M_PI * u.x;
	Float cos_theta = sqrt(u.y);
	Float sin_theta = sqrt(1.0f - u.y);
	return MakeFloat3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

__device__ __forceinline__ Bool SameHemisphere(Float3 w_o, Float3 w_i, Float3 n)
{
	return dot(w_o, n) * dot(w_i, n) > 0.0f;
}

__device__ __forceinline__ Float PowerHeuristic(Float n_f, Float pdf_f, Float n_g, Float pdf_g)
{
	Float f = n_f * pdf_f;
	Float g = n_g * pdf_g;
	return (f * f) / (f * f + g * g + EPSILON);
}

__device__ __forceinline__ void BuildONB(Float3 N, Float3& T, Float3& B)
{
	if (N.z < -0.99998796f)
	{
		T = MakeFloat3(0.0f, -1.0f, 0.0f);
		B = MakeFloat3(-1.0f, 0.0f, 0.0f);
		return;
	}
	Float nxa = -N.x / (1.0f + N.z);
	T = MakeFloat3(1.0f + N.x * nxa, nxa * N.y, -N.x);
	B = MakeFloat3(T.y, 1.0f - N.y * N.y / (1.0f + N.z), -N.y);
}

__device__ __forceinline__ Float3 TangentToWorld(Float3 v, Float3 T, Float3 B, Float3 N)
{
	return v.x * T + v.y * B + v.z * N;
}

__device__ __forceinline__ Float3 WorldToTangent(Float3 v, Float3 T, Float3 B, Float3 N)
{
	return MakeFloat3(dot(v, T), dot(v, B), dot(v, N));
}

__device__ __forceinline__ Float SchlickFresnel(Float VdotH)
{
	Float m = clamp(1.0f - VdotH, 0.0f, 1.0f);
	Float m2 = m * m;
	return m2 * m2 * m;
}

__device__ __forceinline__ Float DielectricFresnel(Float VdotH, Float Eta)
{
	Float cosThetaI = VdotH;
	Float sinThetaTSq = Eta * Eta * (1.0f - cosThetaI * cosThetaI);
	if (sinThetaTSq > 1.0f)
		return 1.0f;
	Float cosThetaT = sqrt(max(1.0f - sinThetaTSq, 0.0f));
	Float rs = (Eta * cosThetaT - cosThetaI) / (Eta * cosThetaT + cosThetaI);
	Float rp = (Eta * cosThetaI - cosThetaT) / (Eta * cosThetaI + cosThetaT);
	return 0.5f * (rs * rs + rp * rp);
}

__device__ __forceinline__ Float GGXDistributionAnisotropic(Float3 H, Float Ax, Float Ay)
{
	Float Hx2 = H.x * H.x;
	Float Hy2 = H.y * H.y;
	Float Hz2 = H.z * H.z;
	Float ax2 = Ax * Ax;
	Float ay2 = Ay * Ay;
	Float denom = Hx2 / ax2 + Hy2 / ay2 + Hz2;
	return 1.0f / (M_PI * Ax * Ay * denom * denom);
}

__device__ __forceinline__ Float Lambda(Float3 V, Float Ax, Float Ay)
{
	Float Vx2 = V.x * V.x;
	Float Vy2 = V.y * V.y;
	Float Vz2 = V.z * V.z;
	Float ax2 = Ax * Ax;
	Float ay2 = Ay * Ay;
	Float numerator = -1.0f + sqrt(1.0f + (ax2 * Vx2 + ay2 * Vy2) / Vz2);
	return numerator / 2.0f;
}

__device__ __forceinline__ Float GGXSmithAnisotropic(Float3 V, Float Ax, Float Ay)
{
	return 1.0f / (1.0f + Lambda(V, Ax, Ay));
}

__device__ __forceinline__ Float3 SampleGGXVNDF(Float3 Ve, Float Ax, Float Ay, Float2 u)
{
	Float3 Vh = normalize(MakeFloat3(Ax * Ve.x, Ay * Ve.y, Ve.z));
	Float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
	Float3 T1 = lensq > 0.0f ? MakeFloat3(-Vh.y, Vh.x, 0.0f) * rsqrt(lensq) : MakeFloat3(1.0f, 0.0f, 0.0f);
	Float3 T2 = cross(Vh, T1);
	Float r = sqrt(u.x);
	Float phi = 2.0f * M_PI * u.y;
	Float t1 = r * cos(phi);
	Float t2 = r * sin(phi);
	Float s = 0.5f * (1.0f + Vh.z);
	t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;
	Float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
	Float3 Ne = normalize(MakeFloat3(Ax * Nh.x, Ay * Nh.y, max(0.0f, Nh.z)));
	return Ne;
}

__device__ __forceinline__ BxDFEval EvaluateDiffuse(Float3 V, Float3 L, Float3 baseColor)
{
	BxDFEval eval;
	if (L.z <= 0.0f)
	{
		eval.PDF = 0.0f;
		eval.BxDF = MakeFloat3(0.0f);
		return eval;
	}
	eval.PDF = L.z * INV_PI;
	eval.BxDF = INV_PI * baseColor;
	return eval;
}

__device__ __forceinline__ BxDFEval EvaluateReflection(Float3 V, Float3 L, Float3 F, Float Ax, Float Ay)
{
	BxDFEval eval;
	if (L.z <= EPSILON)
	{
		eval.BxDF = MakeFloat3(0.0f);
		eval.PDF = 0.0f;
		return eval;
	}
	Float3 H = normalize(V + L);
	Float VdotH = dot(V, H);
	Float D = GGXDistributionAnisotropic(H, Ax, Ay);
	Float GV = GGXSmithAnisotropic(V, Ax, Ay);
	Float GL = GGXSmithAnisotropic(L, Ax, Ay);
	eval.PDF = (GV * max(VdotH, 0.0f) * D / V.z) / (4.0f * VdotH);
	eval.BxDF = D * F * GV * GL / (4.0f * V.z);
	return eval;
}

__device__ __forceinline__ BxDFEval EvaluateRefraction(Float3 V, Float3 L, Float3 F, Float Ax, Float Ay, Float Eta)
{
	BxDFEval eval;
	if (L.z >= -EPSILON)
	{
		eval.BxDF = MakeFloat3(0.0f);
		eval.PDF = 0.0f;
		return eval;
	}
	Float3 H = normalize(Eta * V + L);
	if (H.z < 0.0f)
		H = -H;
	Float VdotH = dot(V, H);
	Float LdotH = dot(L, H);
	Float D = GGXDistributionAnisotropic(H, Ax, Ay);
	Float GV = GGXSmithAnisotropic(V, Ax, Ay);
	Float GL = GGXSmithAnisotropic(L, Ax, Ay);
	Float denominator = (LdotH + Eta * VdotH);
	Float denominator2 = denominator * denominator;
	Float eta2 = Eta * Eta;
	Float jacobian = (eta2 * fabs(LdotH)) / denominator2;
	eval.PDF = (GV * fabs(VdotH) * D / V.z) * jacobian;
	eval.BxDF = (F * D * GV * GL * eta2 / denominator2) * (fabs(VdotH) * fabs(LdotH) / fabs(V.z));
	return eval;
}

__device__ __forceinline__ BxDFEval EvaluateMetallic(Float3 V, Float3 L, Float3 baseColor, Float3 specularColor, Float Ax, Float Ay)
{
	Float3 H = normalize(V + L);
	Float3 F = lerp(baseColor, specularColor, SchlickFresnel(dot(V, H)));
	return EvaluateReflection(V, L, F, Ax, Ay);
}

__device__ __forceinline__ BxDFEval EvaluateDielectricReflection(Float3 V, Float3 L, Float3 specularColor, Float Ax, Float Ay)
{
	return EvaluateReflection(V, L, specularColor, Ax, Ay);
}

__device__ __forceinline__ BxDFEval EvaluateBSDF(EvaluatedMaterial const& mat, Float3 V, Float3 L)
{
	Float metallicProb = mat.metallic;
	Float dielectricProb = (1.0f - mat.metallic) * (1.0f - mat.transmission);
	Float glassProb = (1.0f - mat.metallic) * mat.transmission;

	Float probSum = metallicProb + dielectricProb + glassProb;
	if (probSum > EPSILON)
	{
		metallicProb /= probSum;
		dielectricProb /= probSum;
		glassProb /= probSum;
	}

	Bool refracted = L.z < 0.0f;
	Float3 H;
	Bool validRefraction = false;

	if (refracted)
	{
		H = normalize(mat.Eta * V + L);
		if (H.z < 0.0f)
			H = -H;
		Float VdotH = dot(V, H);
		Float LdotH = dot(L, H);
		validRefraction = (VdotH > 0.0f && LdotH < 0.0f) || (VdotH < 0.0f && LdotH > 0.0f);
	}
	else
	{
		H = normalize(V + L);
	}

	Float FDielectric = DielectricFresnel(fabs(dot(V, H)), mat.Eta);

	BxDFEval directionEval;
	directionEval.BxDF = MakeFloat3(0.0f);
	directionEval.PDF = 0.0f;

	if (!refracted)
	{
		BxDFEval metallicEval = EvaluateMetallic(V, L, mat.base_color, mat.specular_color, mat.Ax, mat.Ay);
		directionEval.BxDF = directionEval.BxDF + metallicEval.BxDF * metallicProb;
		directionEval.PDF += metallicEval.PDF * metallicProb;
	}

	if (!refracted)
	{
		BxDFEval diffuseEval = EvaluateDiffuse(V, L, mat.base_color);
		directionEval.BxDF = directionEval.BxDF + diffuseEval.BxDF * dielectricProb * (1.0f - FDielectric);
		directionEval.PDF += diffuseEval.PDF * dielectricProb * (1.0f - FDielectric);
	}

	if (!refracted)
	{
		BxDFEval specularEval = EvaluateDielectricReflection(V, L, mat.specular_color, mat.Ax, mat.Ay);
		directionEval.BxDF = directionEval.BxDF + specularEval.BxDF * dielectricProb * FDielectric;
		directionEval.PDF += specularEval.PDF * dielectricProb * FDielectric;
	}

	if (!refracted)
	{
		BxDFEval glassEval = EvaluateReflection(V, L, mat.specular_color, mat.Ax, mat.Ay);
		directionEval.BxDF = directionEval.BxDF + glassEval.BxDF * glassProb * FDielectric;
		directionEval.PDF += glassEval.PDF * glassProb * FDielectric;
	}

	if (refracted && validRefraction)
	{
		BxDFEval glassEval = EvaluateRefraction(V, L, mat.base_color, mat.Ax, mat.Ay, mat.Eta);
		directionEval.BxDF = directionEval.BxDF + glassEval.BxDF * glassProb * (1.0f - FDielectric);
		directionEval.PDF += glassEval.PDF * glassProb * (1.0f - FDielectric);
	}

	return directionEval;
}

__device__ __forceinline__ BxDFSample SampleBSDF(EvaluatedMaterial const& mat, Float3 V, PRNG& prng, BSDFComponent& sampledComponent)
{
	Float metallicProb = mat.metallic;
	Float dielectricProb = (1.0f - mat.metallic) * (1.0f - mat.transmission);
	Float glassProb = (1.0f - mat.metallic) * mat.transmission;

	Float probSum = metallicProb + dielectricProb + glassProb;
	if (probSum > EPSILON)
	{
		metallicProb /= probSum;
		dielectricProb /= probSum;
		glassProb /= probSum;
	}

	Float2 u = prng.RandomFloat2();
	Float3 H = SampleGGXVNDF(V, mat.Ax, mat.Ay, u);
	Float FDielectric = DielectricFresnel(dot(V, H), mat.Eta);
	Float x1 = prng.RandomFloat();

	Float3 L;
	Bool refracted = false;
	if (x1 < metallicProb)
	{
		L = normalize(reflect(-V, H));
		sampledComponent = BSDFComponent::Metallic;
	}
	else if (x1 < metallicProb + dielectricProb)
	{
		if (prng.RandomFloat() < FDielectric)
		{
			L = normalize(reflect(-V, H));
			sampledComponent = BSDFComponent::SpecularDielectric;
		}
		else
		{
			Float2 u2 = prng.RandomFloat2();
			L = CosSampleHemisphere(u2);
			sampledComponent = BSDFComponent::Diffuse;
		}
	}
	else
	{
		if (prng.RandomFloat() < FDielectric)
		{
			L = normalize(reflect(-V, H));
			sampledComponent = BSDFComponent::GlassReflect;
		}
		else
		{
			L = normalize(refract(-V, H, mat.Eta));
			sampledComponent = BSDFComponent::GlassRefract;
			refracted = true;
		}
	}

	BxDFSample sample;
	if (L.z < 0.0f && !refracted)
	{
		sample.L = MakeFloat3(0.0f);
		sample.BxDF = MakeFloat3(0.0f);
		sample.PDF = 0.0f;
		return sample;
	}
	else if (refracted && L.z >= 0.0f)
	{
		sample.L = MakeFloat3(0.0f);
		sample.BxDF = MakeFloat3(0.0f);
		sample.PDF = 0.0f;
		return sample;
	}

	BxDFEval scatteringEval = EvaluateBSDF(mat, V, L);
	sample.L = L;
	sample.BxDF = scatteringEval.BxDF;
	sample.PDF = scatteringEval.PDF;

	return sample;
}

__device__ __forceinline__ EvaluatedMaterial EvaluateMaterial(MaterialGPU const& material, Float2 uv, Bool hitFromInside)
{
	EvaluatedMaterial eval;
	eval.base_color = material.base_color;
	if (material.diffuse_tex_id >= 0)
	{
		Float4 sampled = tex2D<Float4>(params.textures[material.diffuse_tex_id], uv.x, uv.y);
		eval.base_color = eval.base_color * MakeFloat3(sampled.x, sampled.y, sampled.z);
	}

	eval.emissive = material.emissive_color;
	if (material.emissive_tex_id >= 0)
	{
		Float4 sampled = tex2D<Float4>(params.textures[material.emissive_tex_id], uv.x, uv.y);
		eval.emissive = eval.emissive * MakeFloat3(sampled.x, sampled.y, sampled.z);
	}

	eval.metallic = material.metallic;
	eval.roughness = material.roughness;
	if (material.metallic_roughness_tex_id >= 0)
	{
		Float4 mr = tex2D<Float4>(params.textures[material.metallic_roughness_tex_id], uv.x, uv.y);
		eval.roughness *= mr.y;
		eval.metallic *= mr.z;
	}

	eval.transmission = material.specular_transmission;
	eval.ior = max(material.ior, 1.000001f);
	eval.anisotropy = material.anisotropy;
	eval.specular_color = MakeFloat3(1.0f);

	Float aspect = sqrt(1.0f - sqrt(eval.anisotropy) * 0.9f);
	eval.Ax = max(0.00001f, eval.roughness / aspect);
	eval.Ay = max(0.00001f, eval.roughness * aspect);

	if (hitFromInside)
		eval.Eta = eval.ior;
	else
		eval.Eta = 1.0f / eval.ior;

	return eval;
}

__device__ __forceinline__ Float3 SampleDirectLight(
	EvaluatedMaterial const& mat,
	Float3 hit_point,
	Float3 V_world,
	Float3 T,
	Float3 B,
	Float3 N,
	PRNG& prng)
{
	if (params.light_count == 0)
		return MakeFloat3(0.0f);

	Uint32 light_index = Uint32(prng.RandomFloat() * Float(params.light_count));
	light_index = min(light_index, params.light_count - 1u);
	LightGPU light = params.lights[light_index];

	Float3 radiance = MakeFloat3(0.0f);
	Float3 V = WorldToTangent(V_world, T, B, N);

	if (light.type == LightGPUType_Directional)
	{
		Float3 light_direction = normalize(light.direction);
		Float3 L_world = -light_direction;
		Float3 L = WorldToTangent(L_world, T, B, N);

		if (L.z > 0.0f)
		{
			if (!TraceOcclusion(params.traversable, hit_point + N * EPSILON, L_world, EPSILON, M_INF))
			{
				BxDFEval bsdf_eval = EvaluateBSDF(mat, V, L);
				radiance = bsdf_eval.BxDF * L.z * light.color * Float(params.light_count);
			}
		}
	}
	else if (light.type == LightGPUType_Point)
	{
		Float3 light_pos = light.position;
		Float3 to_light = light_pos - hit_point;
		Float dist = length(to_light);
		Float3 L_world = to_light / dist;
		Float3 L = WorldToTangent(L_world, T, B, N);

		if (L.z > 0.0f)
		{
			if (!TraceOcclusion(params.traversable, hit_point + N * EPSILON, L_world, EPSILON, dist - EPSILON))
			{
				Float attenuation = 1.0f / (dist * dist);
				BxDFEval bsdf_eval = EvaluateBSDF(mat, V, L);
				radiance = bsdf_eval.BxDF * L.z * light.color * attenuation * Float(params.light_count);
			}
		}
	}

	return radiance;
}

struct HitVertex
{
	Float3 P;
	Float3 Ng;
	Float3 Ns;
	Float2 texcoord;
};

__device__ __forceinline__ HitVertex LoadHitVertex(MeshGPU const& mesh, Uint32 primitive_idx, Float2 barycentrics)
{
	HitVertex vtx;

	Uint3* mesh_indices = params.indices + mesh.indices_offset;
	Uint3 primitive_indices = mesh_indices[primitive_idx];
	Uint32 i0 = primitive_indices.x;
	Uint32 i1 = primitive_indices.y;
	Uint32 i2 = primitive_indices.z;
	Float w = 1.0f - barycentrics.x - barycentrics.y;

	Float3* mesh_vertices = params.vertices + mesh.positions_offset;
	Float3 pos0 = mesh_vertices[i0];
	Float3 pos1 = mesh_vertices[i1];
	Float3 pos2 = mesh_vertices[i2];
	vtx.P = pos0 * w + pos1 * barycentrics.x + pos2 * barycentrics.y;

	Float3 edge1 = pos1 - pos0;
	Float3 edge2 = pos2 - pos0;
	vtx.Ng = normalize(cross(edge1, edge2));

	Float3* mesh_normals = params.normals + mesh.normals_offset;
	Float3 nor0 = mesh_normals[i0];
	Float3 nor1 = mesh_normals[i1];
	Float3 nor2 = mesh_normals[i2];
	vtx.Ns = normalize(nor0 * w + nor1 * barycentrics.x + nor2 * barycentrics.y);

	if (isnan(vtx.Ns.x) || isnan(vtx.Ns.y) || isnan(vtx.Ns.z) || length(vtx.Ns) < 0.5f)
		vtx.Ns = vtx.Ng;

	Float2* mesh_uvs = params.uvs + mesh.uvs_offset;
	Float2 uv0 = mesh_uvs[i0];
	Float2 uv1 = mesh_uvs[i1];
	Float2 uv2 = mesh_uvs[i2];
	vtx.texcoord = uv0 * w + uv1 * barycentrics.x + uv2 * barycentrics.y;
	vtx.texcoord.y = 1.0f - vtx.texcoord.y;

	return vtx;
}

__device__ __forceinline__ Float3 TransformVertex(Float const matrix[12], Float3 const& position)
{
	Float3 transformed;
	transformed.x = matrix[0] * position.x + matrix[1] * position.y + matrix[2] * position.z + matrix[3];
	transformed.y = matrix[4] * position.x + matrix[5] * position.y + matrix[6] * position.z + matrix[7];
	transformed.z = matrix[8] * position.x + matrix[9] * position.y + matrix[10] * position.z + matrix[11];
	return transformed;
}

__device__ __forceinline__ Float3 TransformNormal(Float const matrix[12], Float3 const& normal)
{
	Float3 transformed;
	transformed.x = matrix[0] * normal.x + matrix[1] * normal.y + matrix[2] * normal.z;
	transformed.y = matrix[4] * normal.x + matrix[5] * normal.y + matrix[6] * normal.z;
	transformed.z = matrix[8] * normal.x + matrix[9] * normal.y + matrix[10] * normal.z;
	return normalize(transformed);
}

__device__ __forceinline__ void WriteToDenoiserBuffers(Uint32 idx, Float3 const& albedo, Float3 const& normal)
{
	if (params.denoiser_albedo != NULL)
		params.denoiser_albedo[idx] = albedo;

	if (params.denoiser_normals != NULL)
	{
		Float3 view_normal;
		view_normal.x = dot(normal, params.cam_u);
		view_normal.y = dot(normal, params.cam_v);
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
	if (params.output_type == PathTracerOutputGPU_MaterialID)
	{
		Float3 material_id_color = MakeFloat3(
			(material_id * 37) % 255 / 255.0f,
			(material_id * 59) % 255 / 255.0f,
			(material_id * 97) % 255 / 255.0f);
		params.debug_buffer[idx] = material_id_color;
		return;
	}
}

extern "C" __global__ void RG_NAME(rg)()
{
	Uint2 pixel = MakeUint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
	Uint2 screen = MakeUint2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y);
	Uint32 pixel_idx = pixel.x + pixel.y * screen.x;

	Float3 radiance = MakeFloat3(0.0f);

	for (Uint32 sample_idx = 0; sample_idx < params.sample_count; ++sample_idx)
	{
		PRNG prng = PRNG::Create(pixel_idx, sample_idx + params.frame_index);
		Float2 pixel_offset = prng.RandomFloat2();
		Float2 uv = (MakeFloat2(pixel) + pixel_offset) / MakeFloat2(screen);
		uv = uv * 2.0f - 1.0f;
		uv.y = -uv.y;

		Float aspect = params.cam_aspect_ratio;
		Float tan_fov = tan(params.cam_fovy * 0.5f);

		Float3 ray_origin = params.cam_eye;
		Float3 ray_direction = normalize(
			params.cam_u * uv.x * aspect * tan_fov +
			params.cam_v * uv.y * tan_fov +
			params.cam_w);

		Float3 throughput = MakeFloat3(1.0f);

		for (Uint32 depth = 0; depth < params.max_depth; ++depth)
		{
			HitRecord hit_record{};
			Uint32 p0 = PackPointer0(&hit_record), p1 = PackPointer1(&hit_record);
			Trace(params.traversable, ray_origin, ray_direction, EPSILON, M_INF, p0, p1);

			if (!hit_record.hit)
			{
				Float2 sky_uv = MakeFloat2(
					(1.0f + atan2(ray_direction.x, -ray_direction.z) * INV_PI) * 0.5f,
					1.0f - acos(ray_direction.y) * INV_PI);
				Float3 sky_color = MakeFloat3(0.0f);
				if (params.sky)
				{
					Float4 sampled = tex2D<Float4>(params.sky, sky_uv.x, sky_uv.y);
					sky_color = MakeFloat3(sampled.x, sampled.y, sampled.z);
				}
				radiance += throughput * sky_color;

				if (depth == 0)
				{
					WriteToDenoiserBuffers(pixel_idx, MakeFloat3(0.0f), MakeFloat3(0.0f));
					WriteToDebugBuffer(pixel_idx, MakeFloat3(0.0f), MakeFloat3(0.0f), MakeFloat2(0.0f), 0);
				}
				break;
			}

			MeshGPU mesh = params.meshes[hit_record.instance_idx];
			MaterialGPU material_gpu = params.materials[mesh.material_idx];
			Float3 hit_point = hit_record.P;
			Float3 w_o = -ray_direction;

			Bool hitFromInside = dot(w_o, hit_record.Ns) < 0.0f;

			Float3 Ns = hit_record.Ns;
			Float3 Ng = hit_record.Ng;
			if (material_gpu.specular_transmission == 0.0f && hitFromInside)
			{
				Ns = -Ns;
				Ng = -Ng;
			}

			Float3 T, B;
			BuildONB(Ns, T, B);

			EvaluatedMaterial mat = EvaluateMaterial(material_gpu, hit_record.uv, hitFromInside);

			if (depth == 0)
			{
				WriteToDenoiserBuffers(pixel_idx, mat.base_color, Ns);
				WriteToDebugBuffer(pixel_idx, mat.base_color, Ng, hit_record.uv, mesh.material_idx);
			}

			radiance += throughput * mat.emissive;
			radiance += throughput * SampleDirectLight(mat, hit_point, w_o, T, B, Ns, prng);

			Float3 V = WorldToTangent(w_o, T, B, Ns);
			BSDFComponent sampledComponent;
			BxDFSample bsdf_sample = SampleBSDF(mat, V, prng, sampledComponent);

			if (bsdf_sample.PDF < EPSILON || length(bsdf_sample.BxDF) < EPSILON)
				break;

			Float3 w_i = TangentToWorld(bsdf_sample.L, T, B, Ns);
			throughput = throughput * bsdf_sample.BxDF * fabs(bsdf_sample.L.z) / bsdf_sample.PDF;

			Float3 offset_normal = (bsdf_sample.L.z > 0.0f) ? Ns : -Ns;
			ray_origin = hit_point + offset_normal * EPSILON;
			ray_direction = w_i;

			if (depth >= 2)
			{
				Float q = min(max(throughput.x, max(throughput.y, throughput.z)) + 0.001f, 0.95f);
				if (prng.RandomFloat() > q)
					break;
				throughput = throughput / q;
			}
		}
	}

	radiance = radiance / Float(params.sample_count);

	Float luminance = dot(radiance, MakeFloat3(0.2126f, 0.7152f, 0.0722f));
	if (luminance > 50.0f)
		radiance = radiance * 50.0f / luminance;

	Float3 accumulated;
	if (params.frame_index > 0)
	{
		Float3 old_accum = params.accum_buffer[pixel_idx];
		accumulated = old_accum + radiance;
	}
	else
	{
		accumulated = radiance;
	}
	params.accum_buffer[pixel_idx] = accumulated;
}

extern "C" __global__ void MISS_NAME(ms)()
{
	GetPayload<HitRecord>()->hit = false;
}

extern "C" __global__ void AH_NAME(ah)()
{
	Uint32 instance_idx = optixGetInstanceId();
	Uint32 primitive_idx = optixGetPrimitiveIndex();

	MeshGPU mesh = params.meshes[instance_idx];
	MaterialGPU material = params.materials[mesh.material_idx];
	if (material.diffuse_tex_id >= 0)
	{
		HitVertex vtx = LoadHitVertex(mesh, primitive_idx, optixGetTriangleBarycentrics());
		Float4 sampled = tex2D<Float4>(params.textures[material.diffuse_tex_id], vtx.texcoord.x, vtx.texcoord.y);
		if (sampled.w < material.alpha_cutoff)
			optixIgnoreIntersection();
	}
}

extern "C" __global__ void CH_NAME(ch)()
{
	Uint32 instance_idx = optixGetInstanceId();
	Uint32 primitive_idx = optixGetPrimitiveIndex();
	Float2 barycentrics = optixGetTriangleBarycentrics();

	MeshGPU mesh = params.meshes[instance_idx];
	HitVertex vtx = LoadHitVertex(mesh, primitive_idx, barycentrics);

	Float object_to_world[12];
	optixGetObjectToWorldTransformMatrix(object_to_world);

	HitRecord* hit_record = GetPayload<HitRecord>();
	hit_record->P = TransformVertex(object_to_world, vtx.P);
	hit_record->Ng = TransformNormal(object_to_world, vtx.Ng);
	hit_record->Ns = TransformNormal(object_to_world, vtx.Ns);
	hit_record->uv = vtx.texcoord;
	hit_record->barycentrics = barycentrics;
	hit_record->primitive_idx = primitive_idx;
	hit_record->instance_idx = instance_idx;
	hit_record->hit = true;
	hit_record->t = optixGetRayTmax();
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
		HitVertex vtx = LoadHitVertex(mesh, primitive_idx, barycentrics);
		Float4 sampled = tex2D<Float4>(params.textures[material.diffuse_tex_id], vtx.texcoord.x, vtx.texcoord.y);
		if (sampled.w < material.alpha_cutoff)
			optixIgnoreIntersection();
	}
	if (material.specular_transmission > 0)
		optixIgnoreIntersection();
}
