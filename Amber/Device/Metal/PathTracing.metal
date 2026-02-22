#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

#include "MetalDeviceHostCommon.h"

using namespace amber;

constant float INV_PI = 0.31830988618379067154f;
constant float EPSILON = 1e-5f;

struct PRNG
{
    uint seed;
    static PRNG Create(uint pixel_idx, uint frame)
    {
        PRNG prng;
        prng.seed = pixel_idx + frame * 719393u;
        return prng;
    }

    float RandomFloat()
    {
        seed = (seed ^ 61u) ^ (seed >> 16u);
        seed *= 9u;
        seed = seed ^ (seed >> 4u);
        seed *= 0x27d4eb2du;
        seed = seed ^ (seed >> 15u);
        return float(seed) / 4294967296.0f;
    }

    float2 RandomFloat2()
    {
        return float2(RandomFloat(), RandomFloat());
    }

    float3 RandomFloat3()
    {
        return float3(RandomFloat(), RandomFloat(), RandomFloat());
    }
};

struct BxDFSample
{
    float3 L;       // Sampled direction (tangent space)
    float3 BxDF;    // BRDF/BTDF value
    float PDF;      // Probability density
};

struct BxDFEval
{
    float3 BxDF;
    float PDF;
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
    float3 base_color;
    float3 emissive;
    float3 specular_color;
    float metallic;
    float roughness;
    float transmission;
    float ior;
    float anisotropy;
    float Ax;
    float Ay;
    float Eta;
};

float Pow2(float x)
{
    return x * x;
}

float3 CosSampleHemisphere(float2 u)
{
    float phi = 2.0f * M_PI_F * u.x;
    float cos_theta = sqrt(u.y);
    float sin_theta = sqrt(1.0f - u.y);
    return float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

bool SameHemisphere(float3 w_o, float3 w_i, float3 n)
{
    return dot(w_o, n) * dot(w_i, n) > 0.0f;
}

float PowerHeuristic(float n_f, float pdf_f, float n_g, float pdf_g)
{
    float f = n_f * pdf_f;
    float g = n_g * pdf_g;
    return (f * f) / (f * f + g * g + EPSILON);
}

void BuildONB(float3 N, thread float3& T, thread float3& B)
{
    if (N.z < -0.99998796f)
    {
        T = float3(0.0f, -1.0f, 0.0f);
        B = float3(-1.0f, 0.0f, 0.0f);
        return;
    }
    float nxa = -N.x / (1.0f + N.z);
    T = float3(1.0f + N.x * nxa, nxa * N.y, -N.x);
    B = float3(T.y, 1.0f - N.y * N.y / (1.0f + N.z), -N.y);
}

float3 TangentToWorld(float3 v, float3 T, float3 B, float3 N)
{
    return v.x * T + v.y * B + v.z * N;
}

float3 WorldToTangent(float3 v, float3 T, float3 B, float3 N)
{
    return float3(dot(v, T), dot(v, B), dot(v, N));
}

float SchlickFresnel(float VdotH)
{
    float m = clamp(1.0f - VdotH, 0.0f, 1.0f);
    float m2 = m * m;
    return m2 * m2 * m;
}

float DielectricFresnel(float VdotH, float Eta)
{
    float cosThetaI = VdotH;
    float sinThetaTSq = Eta * Eta * (1.0f - cosThetaI * cosThetaI);

    if (sinThetaTSq > 1.0f)
        return 1.0f;

    float cosThetaT = sqrt(max(1.0f - sinThetaTSq, 0.0f));

    float rs = (Eta * cosThetaT - cosThetaI) / (Eta * cosThetaT + cosThetaI);
    float rp = (Eta * cosThetaI - cosThetaT) / (Eta * cosThetaI + cosThetaT);

    return 0.5f * (rs * rs + rp * rp);
}

float GGXDistributionAnisotropic(float3 H, float Ax, float Ay)
{
    float Hx2 = H.x * H.x;
    float Hy2 = H.y * H.y;
    float Hz2 = H.z * H.z;

    float ax2 = Ax * Ax;
    float ay2 = Ay * Ay;

    float denom = Hx2 / ax2 + Hy2 / ay2 + Hz2;
    return 1.0f / (M_PI_F * Ax * Ay * denom * denom);
}

float Lambda(float3 V, float Ax, float Ay)
{
    float Vx2 = V.x * V.x;
    float Vy2 = V.y * V.y;
    float Vz2 = V.z * V.z;

    float ax2 = Ax * Ax;
    float ay2 = Ay * Ay;

    float numerator = -1.0f + sqrt(1.0f + (ax2 * Vx2 + ay2 * Vy2) / Vz2);
    return numerator / 2.0f;
}

float GGXSmithAnisotropic(float3 V, float Ax, float Ay)
{
    return 1.0f / (1.0f + Lambda(V, Ax, Ay));
}

float3 SampleGGXVNDF(float3 Ve, float Ax, float Ay, float2 u)
{
    float3 Vh = normalize(float3(Ax * Ve.x, Ay * Ve.y, Ve.z));

    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0f ? float3(-Vh.y, Vh.x, 0.0f) * rsqrt(lensq) : float3(1.0f, 0.0f, 0.0f);
    float3 T2 = cross(Vh, T1);

    float r = sqrt(u.x);
    float phi = 2.0f * M_PI_F * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;

    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
    float3 Ne = normalize(float3(Ax * Nh.x, Ay * Nh.y, max(0.0f, Nh.z)));
    return Ne;
}

BxDFEval EvaluateDiffuse(float3 V, float3 L, float3 baseColor)
{
    BxDFEval eval;

    if (L.z <= 0.0f)
    {
        eval.PDF = 0.0f;
        eval.BxDF = float3(0.0f);
        return eval;
    }

    eval.PDF = L.z * INV_PI;
    eval.BxDF = INV_PI * baseColor;  
    return eval;
}

BxDFEval EvaluateReflection(float3 V, float3 L, float3 F, float Ax, float Ay)
{
    BxDFEval eval;

    if (L.z <= EPSILON)
    {
        eval.BxDF = float3(0.0f);
        eval.PDF = 0.0f;
        return eval;
    }

    float3 H = normalize(V + L);

    float VdotH = dot(V, H);
    float D = GGXDistributionAnisotropic(H, Ax, Ay);
    float GV = GGXSmithAnisotropic(V, Ax, Ay);
    float GL = GGXSmithAnisotropic(L, Ax, Ay);

    eval.PDF = (GV * max(VdotH, 0.0f) * D / V.z) / (4.0f * VdotH);
    eval.BxDF = D * F * GV * GL / (4.0f * V.z);

    return eval;
}

BxDFEval EvaluateRefraction(float3 V, float3 L, float3 F, float Ax, float Ay, float Eta)
{
    BxDFEval eval;

    if (L.z >= -EPSILON)
    {
        eval.BxDF = float3(0.0f);
        eval.PDF = 0.0f;
        return eval;
    }

    float3 H = normalize(Eta * V + L);

    if (H.z < 0.0f)
        H = -H;

    float VdotH = dot(V, H);
    float LdotH = dot(L, H);

    float D = GGXDistributionAnisotropic(H, Ax, Ay);
    float GV = GGXSmithAnisotropic(V, Ax, Ay);
    float GL = GGXSmithAnisotropic(L, Ax, Ay);

    float denominator = (LdotH + Eta * VdotH);
    float denominator2 = denominator * denominator;
    float eta2 = Eta * Eta;

    float jacobian = (eta2 * abs(LdotH)) / denominator2;

    eval.PDF = (GV * abs(VdotH) * D / V.z) * jacobian;
    eval.BxDF = (F * D * GV * GL * eta2 / denominator2) * (abs(VdotH) * abs(LdotH) / abs(V.z));

    return eval;
}

BxDFEval EvaluateMetallic(float3 V, float3 L, float3 baseColor, float3 specularColor, float Ax, float Ay)
{
    float3 H = normalize(V + L);
    float3 F = mix(baseColor, specularColor, SchlickFresnel(dot(V, H)));
    return EvaluateReflection(V, L, F, Ax, Ay);
}

BxDFEval EvaluateDielectricReflection(float3 V, float3 L, float3 specularColor, float Ax, float Ay)
{
    return EvaluateReflection(V, L, specularColor, Ax, Ay);
}

BxDFEval EvaluateBSDF(EvaluatedMaterial mat, float3 V, float3 L)
{
    float metallicProb = mat.metallic;
    float dielectricProb = (1.0f - mat.metallic) * (1.0f - mat.transmission);
    float glassProb = (1.0f - mat.metallic) * mat.transmission;

    float probSum = metallicProb + dielectricProb + glassProb;
    if (probSum > EPSILON)
    {
        metallicProb /= probSum;
        dielectricProb /= probSum;
        glassProb /= probSum;
    }

    bool refracted = L.z < 0.0f;
    float3 H;
    bool validRefraction = false;

    if (refracted)
    {
        H = normalize(mat.Eta * V + L);
        if (H.z < 0.0f)
            H = -H;

        float VdotH = dot(V, H);
        float LdotH = dot(L, H);
        validRefraction = (VdotH > 0.0f && LdotH < 0.0f) || (VdotH < 0.0f && LdotH > 0.0f);
    }
    else
    {
        H = normalize(V + L);
    }

    float FDielectric = DielectricFresnel(abs(dot(V, H)), mat.Eta);

    BxDFEval directionEval;
    directionEval.BxDF = float3(0.0f);
    directionEval.PDF = 0.0f;

    if (!refracted)
    {
        BxDFEval metallicEval = EvaluateMetallic(V, L, mat.base_color, mat.specular_color, mat.Ax, mat.Ay);
        directionEval.BxDF += metallicEval.BxDF * metallicProb;
        directionEval.PDF += metallicEval.PDF * metallicProb;
    }

    if (!refracted)
    {
        BxDFEval diffuseEval = EvaluateDiffuse(V, L, mat.base_color);
        directionEval.BxDF += diffuseEval.BxDF * dielectricProb * (1.0f - FDielectric);
        directionEval.PDF += diffuseEval.PDF * dielectricProb * (1.0f - FDielectric);
    }

    if (!refracted)
    {
        BxDFEval specularEval = EvaluateDielectricReflection(V, L, mat.specular_color, mat.Ax, mat.Ay);
        directionEval.BxDF += specularEval.BxDF * dielectricProb * FDielectric;
        directionEval.PDF += specularEval.PDF * dielectricProb * FDielectric;
    }

    if (!refracted)
    {
        BxDFEval glassEval = EvaluateReflection(V, L, mat.specular_color, mat.Ax, mat.Ay);
        directionEval.BxDF += glassEval.BxDF * glassProb * FDielectric;
        directionEval.PDF += glassEval.PDF * glassProb * FDielectric;
    }

    if (refracted && validRefraction)
    {
        BxDFEval glassEval = EvaluateRefraction(V, L, mat.base_color, mat.Ax, mat.Ay, mat.Eta);
        directionEval.BxDF += glassEval.BxDF * glassProb * (1.0f - FDielectric);
        directionEval.PDF += glassEval.PDF * glassProb * (1.0f - FDielectric);
    }

    return directionEval;
}

BxDFSample SampleBSDF(EvaluatedMaterial mat, float3 V, thread PRNG& prng, thread BSDFComponent& sampledComponent)
{
    float metallicProb = mat.metallic;
    float dielectricProb = (1.0f - mat.metallic) * (1.0f - mat.transmission);
    float glassProb = (1.0f - mat.metallic) * mat.transmission;

    float probSum = metallicProb + dielectricProb + glassProb;
    if (probSum > EPSILON)
    {
        metallicProb /= probSum;
        dielectricProb /= probSum;
        glassProb /= probSum;
    }

    float2 u = prng.RandomFloat2();
    float3 H = SampleGGXVNDF(V, mat.Ax, mat.Ay, u);
    float FDielectric = DielectricFresnel(dot(V, H), mat.Eta);
    float x1 = prng.RandomFloat();

    float3 L;
    bool refracted = false;
    if (x1 < metallicProb)
    {
        L = normalize(reflect(-V, H));
        sampledComponent = BSDFComponent::Metallic;
    }
    else if (x1 < metallicProb + dielectricProb)
    {
        if (prng.RandomFloat() < FDielectric)
        {
            // reflect
            L = normalize(reflect(-V, H));
            sampledComponent = BSDFComponent::SpecularDielectric;
        }
        else
        {
            // diffuse scatter
            float2 u2 = prng.RandomFloat2();
            L = CosSampleHemisphere(u2);
            sampledComponent = BSDFComponent::Diffuse;
        }
    }
    else
    {
        // sample glass lobe
        if (prng.RandomFloat() < FDielectric)
        {
            // reflect
            L = normalize(reflect(-V, H));
            sampledComponent = BSDFComponent::GlassReflect;
        }
        else
        {
            // refract
            L = normalize(refract(-V, H, mat.Eta));
            sampledComponent = BSDFComponent::GlassRefract;
            refracted = true;
        }
    }
    BxDFSample sample;
    if (L.z < 0.0f && !refracted)
    {
        sample.L = float3(0.0f);
        sample.BxDF = float3(0.0f);
        sample.PDF = 0.0f;
        return sample;
    }
    else if (refracted && L.z >= 0.0f)
    {
        sample.L = float3(0.0f);
        sample.BxDF = float3(0.0f);
        sample.PDF = 0.0f;
        return sample;
    }

    BxDFEval scatteringEval = EvaluateBSDF(mat, V, L);
    sample.L = L;
    sample.BxDF = scatteringEval.BxDF;
    sample.PDF = scatteringEval.PDF;

    return sample;
}

EvaluatedMaterial EvaluateMaterial(
    constant MaterialGPU& material,
    float2 uv,
    constant SceneResources& scene,
    bool hitFromInside)
{
    EvaluatedMaterial eval;
    eval.base_color = material.base_color.rgb;
    if (material.diffuse_tex_id >= 0)
    {
        constexpr sampler tex_sampler(filter::linear, address::repeat);
        float3 tex_color = scene.textures[material.diffuse_tex_id].sample(tex_sampler, uv).rgb;
        eval.base_color *= tex_color;
    }

    eval.emissive = material.emissive_color.rgb;
    if (material.emissive_tex_id >= 0)
    {
        constexpr sampler tex_sampler(filter::linear, address::repeat);
        float3 tex_emissive = scene.textures[material.emissive_tex_id].sample(tex_sampler, uv).rgb;
        eval.emissive *= tex_emissive;
    }

    eval.metallic = material.metallic;
    eval.roughness = material.roughness;
    if (material.metallic_roughness_tex_id >= 0)
    {
        constexpr sampler tex_sampler(filter::linear, address::repeat);
        float4 mr = scene.textures[material.metallic_roughness_tex_id].sample(tex_sampler, uv);
        eval.roughness *= mr.g;
        eval.metallic *= mr.b;
    }

    eval.transmission = material.specular_transmission;
    eval.ior = max(material.ior, 1.000001f);
    eval.anisotropy = material.anisotropy;

    eval.specular_color = float3(1.0f);

    float aspect = sqrt(1.0f - sqrt(eval.anisotropy) * 0.9f);
    eval.Ax = max(0.00001f, eval.roughness / aspect);
    eval.Ay = max(0.00001f, eval.roughness * aspect);

    if (hitFromInside)
    {
        eval.Eta = eval.ior;
    }
    else
    {
        eval.Eta = 1.0f / eval.ior;
    }

    return eval;
}

float3 SampleDirectLight(
    EvaluatedMaterial mat,
    float3 hit_point,
    float3 V_world,
    float3 T,
    float3 B,
    float3 N,
    thread PRNG& prng,
    constant RenderParams& params,
    constant SceneResources& scene,
    instance_acceleration_structure accel_structure)
{
    if (params.light_count == 0)
        return float3(0.0f);

    uint light_index = uint(prng.RandomFloat() * float(params.light_count));
    light_index = min(light_index, params.light_count - 1u);
    constant LightGPU& light = scene.lights[light_index];

    float3 radiance = float3(0.0f);

    float3 V = WorldToTangent(V_world, T, B, N);
    if (light.type == LightGPUType_Directional)
    {
        float3 light_direction = normalize(light.direction.xyz);
        float3 L_world = -light_direction;
        float3 L = WorldToTangent(L_world, T, B, N);

        if (L.z > 0.0f)
        {
            ray shadow_ray;
            shadow_ray.origin = hit_point + N * EPSILON;
            shadow_ray.direction = L_world;
            shadow_ray.min_distance = EPSILON;
            shadow_ray.max_distance = INFINITY;

            intersector<triangle_data, instancing> shadow_intersect;
            shadow_intersect.accept_any_intersection(true);
            shadow_intersect.assume_geometry_type(geometry_type::triangle);
            auto shadow_result = shadow_intersect.intersect(shadow_ray, accel_structure);
            if (shadow_result.type == intersection_type::none)
            {
                BxDFEval bsdf_eval = EvaluateBSDF(mat, V, L);
                radiance = bsdf_eval.BxDF * L.z * light.color.rgb * float(params.light_count);
            }
        }
    }
    else if (light.type == LightGPUType_Point)
    {
        float3 light_pos = light.position.xyz;
        float3 to_light = light_pos - hit_point;
        float dist = length(to_light);
        float3 L_world = to_light / dist;
        float3 L = WorldToTangent(L_world, T, B, N);
        if (L.z > 0.0f)
        {
            ray shadow_ray;
            shadow_ray.origin = hit_point + N * EPSILON;
            shadow_ray.direction = L_world;
            shadow_ray.min_distance = EPSILON;
            shadow_ray.max_distance = dist - EPSILON;

            intersector<triangle_data, instancing> shadow_intersect;
            shadow_intersect.accept_any_intersection(true);
            shadow_intersect.assume_geometry_type(geometry_type::triangle);

            auto shadow_result = shadow_intersect.intersect(shadow_ray, accel_structure);
            if (shadow_result.type == intersection_type::none)
            {
                float attenuation = 1.0f / (dist * dist);
                BxDFEval bsdf_eval = EvaluateBSDF(mat, V, L);
                radiance = bsdf_eval.BxDF * L.z * light.color.rgb * attenuation * float(params.light_count);
            }
        }
    }
    return radiance;
}

float3 TransformNormal(constant InstanceData& inst, float3 normal)
{
    float3 transformed;
    transformed.x = inst.transform_row0.x * normal.x + inst.transform_row0.y * normal.y + inst.transform_row0.z * normal.z;
    transformed.y = inst.transform_row1.x * normal.x + inst.transform_row1.y * normal.y + inst.transform_row1.z * normal.z;
    transformed.z = inst.transform_row2.x * normal.x + inst.transform_row2.y * normal.y + inst.transform_row2.z * normal.z;
    return normalize(transformed);
}

struct HitVertex
{
    float3 P;
    float3 Ng;
    float3 Ns;
    float2 texcoord;
};

HitVertex LoadHitVertex(
    constant SceneResources& scene,
    constant MeshGPU& mesh,
    uint primitive_idx,
    float2 barycentrics)
{
    HitVertex vtx;

    uint index_offset = (mesh.indices_offset + primitive_idx) * 3;
    uint i0 = scene.indices[index_offset + 0];
    uint i1 = scene.indices[index_offset + 1];
    uint i2 = scene.indices[index_offset + 2];
    float w = 1.0f - barycentrics.x - barycentrics.y;

    float3 pos0 = scene.vertices[mesh.positions_offset + i0];
    float3 pos1 = scene.vertices[mesh.positions_offset + i1];
    float3 pos2 = scene.vertices[mesh.positions_offset + i2];
    vtx.P = pos0 * w + pos1 * barycentrics.x + pos2 * barycentrics.y;

    float3 edge1 = pos1 - pos0;
    float3 edge2 = pos2 - pos0;
    vtx.Ng = normalize(cross(edge1, edge2));

    float3 nor0 = scene.normals[mesh.normals_offset + i0];
    float3 nor1 = scene.normals[mesh.normals_offset + i1];
    float3 nor2 = scene.normals[mesh.normals_offset + i2];
    vtx.Ns = normalize(nor0 * w + nor1 * barycentrics.x + nor2 * barycentrics.y);

    if (any(isnan(vtx.Ns)) || length(vtx.Ns) < 0.5f)
    {
        vtx.Ns = vtx.Ng;
    }

    float2 uv0 = scene.uvs[mesh.uvs_offset + i0];
    float2 uv1 = scene.uvs[mesh.uvs_offset + i1];
    float2 uv2 = scene.uvs[mesh.uvs_offset + i2];
    vtx.texcoord = uv0 * w + uv1 * barycentrics.x + uv2 * barycentrics.y;
    vtx.texcoord.y = 1.0f - vtx.texcoord.y;

    return vtx;
}

kernel void pathtrace_kernel(
    uint2 gid [[thread_position_in_grid]],
    constant RenderParams& params [[buffer(0)]],
    constant SceneResources& scene [[buffer(1)]],
    instance_acceleration_structure accel_structure [[buffer(2)]],
    texture2d<float, access::read_write> accum   [[texture(0)]],
    texture2d<float>                     sky      [[texture(1)]],
    texture2d<float, access::write>      debug    [[texture(2)]])
{
    if (gid.x >= params.width || gid.y >= params.height)
        return;

    uint pixel_idx = gid.x + gid.y * params.width;

    bool debug_mode = (params.output_type != OutputTypeGPU_Final);

    float aspect  = params.cam_aspect_ratio;
    float tan_fov = tan(params.cam_fovy * 0.5f);
    float3 ray_origin = params.cam_eye.xyz;
    if (debug_mode)
    {
        float2 uv = (float2(gid) + float2(0.5f)) / float2(params.width, params.height);
        uv = uv * 2.0f - 1.0f;
        uv.y = -uv.y;

        float3 ray_direction = normalize(
            params.cam_u.xyz * uv.x * aspect * tan_fov +
            params.cam_v.xyz * uv.y * tan_fov +
            params.cam_w.xyz);

        ray ray_query;
        ray_query.origin       = ray_origin;
        ray_query.direction    = ray_direction;
        ray_query.min_distance = EPSILON;
        ray_query.max_distance = INFINITY;

        intersector<triangle_data, instancing> isect;
        isect.accept_any_intersection(false);
        isect.assume_geometry_type(geometry_type::triangle);

        auto intersection = isect.intersect(ray_query, accel_structure);

        float3 debug_value = float3(0.0f);
        if (intersection.type != intersection_type::none)
        {
            uint primitive_id   = intersection.primitive_id;
            uint instance_id    = intersection.instance_id;
            float2 barycentrics = intersection.triangle_barycentric_coord;

            constant InstanceData& instance    = scene.instances[instance_id];
            constant MeshGPU&      mesh        = scene.meshes[instance.mesh_id];
            constant MaterialGPU&  material_gpu = scene.materials[mesh.material_idx];
            HitVertex vtx = LoadHitVertex(scene, mesh, primitive_id, barycentrics);
            vtx.Ng = TransformNormal(instance, vtx.Ng);

            bool hitFromInside = dot(-ray_direction, vtx.Ns) < 0.0f;
            EvaluatedMaterial mat = EvaluateMaterial(material_gpu, vtx.texcoord, scene, hitFromInside);

            if (params.output_type == OutputTypeGPU_Albedo)
            {
                debug_value = mat.base_color;
            }
            else if (params.output_type == OutputTypeGPU_Normal)
            {
                debug_value = vtx.Ng * 0.5f + 0.5f;
            }
            else if (params.output_type == OutputTypeGPU_UV)
            {
                debug_value = float3(vtx.texcoord, 0.0f);
            }
            else if (params.output_type == OutputTypeGPU_MaterialID)
            {
                uint mid = mesh.material_idx;
                debug_value = float3(
                    float((mid * 37u) % 255u) / 255.0f,
                    float((mid * 59u) % 255u) / 255.0f,
                    float((mid * 97u) % 255u) / 255.0f);
            }
        }
        debug.write(float4(debug_value, 1.0f), gid);
        return;
    }

    float3 radiance = float3(0.0f);
    for (uint sample_idx = 0; sample_idx < params.sample_count; ++sample_idx)
    {
        PRNG sample_prng = PRNG::Create(pixel_idx, sample_idx + params.frame_index);
        float2 offset = sample_prng.RandomFloat2();
        float2 uv = (float2(gid) + offset) / float2(params.width, params.height);
        uv = uv * 2.0f - 1.0f;
        uv.y = -uv.y;

        float3 ro = params.cam_eye.xyz;
        float3 rd = normalize(
            params.cam_u.xyz * uv.x * aspect * tan_fov +
            params.cam_v.xyz * uv.y * tan_fov +
            params.cam_w.xyz);

        float3 throughput = float3(1.0f);
        for (uint depth = 0; depth < params.max_depth; ++depth)
        {
            ray ray_query;
            ray_query.origin = ro;
            ray_query.direction = rd;
            ray_query.min_distance = EPSILON;
            ray_query.max_distance = INFINITY;

            intersector<triangle_data, instancing> intersect;
            intersect.accept_any_intersection(false);
            intersect.assume_geometry_type(geometry_type::triangle);

            auto intersection = intersect.intersect(ray_query, accel_structure);
            if (intersection.type == intersection_type::none)
            {
                float2 sky_uv = float2(
                    (1.0f + atan2(rd.x, -rd.z) * INV_PI) * 0.5f,
                     1.0f - acos(rd.y) * INV_PI);
                constexpr sampler sky_sampler(filter::linear, address::repeat);
                float3 sky_color = sky.sample(sky_sampler, sky_uv).rgb;
                radiance += throughput * sky_color;
                break;
            }
            uint primitive_id = intersection.primitive_id;
            uint instance_id = intersection.instance_id;
            float2 barycentrics = intersection.triangle_barycentric_coord;

            constant InstanceData& instance = scene.instances[instance_id];
            constant MeshGPU& mesh = scene.meshes[instance.mesh_id];
            constant MaterialGPU& material_gpu = scene.materials[mesh.material_idx];
            HitVertex vtx = LoadHitVertex(scene, mesh, primitive_id, barycentrics);

            vtx.Ns = TransformNormal(instance, vtx.Ns);
            vtx.Ng = TransformNormal(instance, vtx.Ng);
            float3 hit_point = ro + rd * intersection.distance;
            float3 w_o = -rd;

            bool hitFromInside = dot(w_o, vtx.Ns) < 0.0f;

            float3 Ns = vtx.Ns;
            float3 Ng = vtx.Ng;
            if (material_gpu.specular_transmission == 0.0f && hitFromInside)
            {
                Ns = -Ns;
                Ng = -Ng;
            }

            float3 T, B;
            BuildONB(Ns, T, B);

            EvaluatedMaterial mat = EvaluateMaterial(material_gpu, vtx.texcoord, scene, hitFromInside);

            radiance += throughput * mat.emissive;

            radiance += throughput * SampleDirectLight(mat, hit_point, w_o, T, B, Ns, sample_prng, params, scene, accel_structure);

            float3 V = WorldToTangent(w_o, T, B, Ns);
            BSDFComponent sampledComponent;
            BxDFSample bsdf_sample = SampleBSDF(mat, V, sample_prng, sampledComponent);
            if (bsdf_sample.PDF < EPSILON || length(bsdf_sample.BxDF) < EPSILON)
            {
                break;
            }

            float3 w_i = TangentToWorld(bsdf_sample.L, T, B, Ns);

            throughput *= bsdf_sample.BxDF * abs(bsdf_sample.L.z) / bsdf_sample.PDF;

            float3 offset_normal = (bsdf_sample.L.z > 0.0f) ? Ns : -Ns;
            ro = hit_point + offset_normal * EPSILON;
            rd = w_i;

            if (depth >= 2)
            {
                float q = min(max(throughput.r, max(throughput.g, throughput.b)) + 0.001f, 0.95f);
                if (sample_prng.RandomFloat() > q)
                    break;
                throughput /= q;
            }
        }
    }

    radiance /= float(params.sample_count);
    float luminance = dot(radiance, float3(0.2126f, 0.7152f, 0.0722f));
    if (luminance > 50.0f)
    {
        radiance *= 50.0f / luminance;
    }

    float3 accumulated;
    if (params.frame_index > 0)
    {
        float3 old_accum = accum.read(gid).rgb;
        accumulated = old_accum + radiance;
    }
    else
    {
        accumulated = radiance;
    }
    accum.write(float4(accumulated, 1.0f), gid);
}
