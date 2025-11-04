#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

#include "MetalDeviceHostCommon.h"

using namespace amber;

struct RayPayload
{
    float3 radiance;
    float3 attenuation;
    float3 ray_origin;
    float3 ray_direction;
    uint depth;
    bool missed;
};

float random(thread uint& seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return float(seed) / 4294967296.0f;
}

float3 random3(thread uint& seed)
{
    return float3(random(seed), random(seed), random(seed));
}

float3 sample_hemisphere(thread uint& seed, float3 normal)
{
    float3 random_dir = normalize(random3(seed) * 2.0f - 1.0f);
    return normalize(normal + random_dir);
}

kernel void pathtrace_kernel(
    uint2 gid [[thread_position_in_grid]],
    constant RenderParams& params [[buffer(0)]],
    constant SceneResources& scene [[buffer(1)]],
    instance_acceleration_structure accel_structure [[buffer(2)]],
    texture2d<float, access::write> output [[texture(0)]],
    texture2d<float, access::read_write> accum [[texture(1)]],
    texture2d<float> sky [[texture(2)]])
{
    if (gid.x >= params.width || gid.y >= params.height)
        return;

    uint seed = gid.x + gid.y * params.width + params.frame_index * 719393;

    float2 uv = (float2(gid) + random3(seed).xy) / float2(params.width, params.height);
    uv = uv * 2.0f - 1.0f;
    uv.y = -uv.y;

    float aspect = params.cam_aspect_ratio;
    float tan_fov = tan(params.cam_fovy * 0.5f);

    float3 ray_origin = params.cam_eye.xyz;
    float3 ray_direction = normalize(
        params.cam_u.xyz * uv.x * aspect * tan_fov +
        params.cam_v.xyz * uv.y * tan_fov +
        params.cam_w.xyz);

    float3 radiance = float3(0.0f);
    float3 throughput = float3(1.0f);

    for (uint depth = 0; depth < params.max_depth; ++depth)
    {
        ray ray_query;
        ray_query.origin = ray_origin;
        ray_query.direction = ray_direction;
        ray_query.min_distance = 0.001f;
        ray_query.max_distance = 10000.0f;

        intersector<triangle_data, instancing> intersect;
        intersect.accept_any_intersection(false);
        intersect.assume_geometry_type(geometry_type::triangle);

        typename intersector<triangle_data, instancing>::result_type intersection =
            intersect.intersect(ray_query, accel_structure);

        if (intersection.type == intersection_type::none)
        {
            float2 sky_uv = float2(
                atan2(ray_direction.z, ray_direction.x) / (2.0f * M_PI_F) + 0.5f,
                1.0f - acos(ray_direction.y) / M_PI_F);
            float3 sky_color = sky.sample(sampler(filter::linear), sky_uv).rgb;
            radiance += throughput * sky_color;
            break;
        }

        uint primitive_id = intersection.primitive_id;
        uint instance_id = intersection.instance_id;

        constant MeshGPU& mesh = scene.meshes[instance_id];
        constant MaterialGPU& material = scene.materials[mesh.material_idx];
        float2 barycentrics = intersection.triangle_barycentric_coord;

        uint triangle_index = mesh.indices_offset + primitive_id;
        uint index_offset = triangle_index * 3;
        uint i0 = scene.indices[index_offset + 0];
        uint i1 = scene.indices[index_offset + 1];
        uint i2 = scene.indices[index_offset + 2];

        float2 uv0 = scene.uvs[mesh.uvs_offset + i0];
        float2 uv1 = scene.uvs[mesh.uvs_offset + i1];
        float2 uv2 = scene.uvs[mesh.uvs_offset + i2];

        float2 tex_uv = uv0 * (1.0f - barycentrics.x - barycentrics.y) +
                        uv1 * barycentrics.x +
                        uv2 * barycentrics.y;
        tex_uv.y = 1.0f - tex_uv.y;

        float3 albedo = material.base_color.rgb;
        if (material.diffuse_tex_id >= 0)
        {
            constexpr sampler tex_sampler(filter::linear, address::repeat);
            float3 tex_color = scene.textures[material.diffuse_tex_id].sample(tex_sampler, tex_uv).rgb;
            albedo *= tex_color;
        }
        radiance = albedo;
        break;
    }

    float3 accumulated = radiance;
    accum.write(float4(accumulated, 1.0f), gid);

    float3 color = accumulated / (accumulated + 1.0f);
    color = pow(color, float3(1.0f / 2.2f));
    output.write(float4(color, 1.0f), gid);
}
