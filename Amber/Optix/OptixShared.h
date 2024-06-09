#pragma once
#include <vector_types.h>
#include <optix_types.h>

#define RG_NAME(name) __raygen__##name
#define MISS_NAME(name) __miss__##name
#define CH_NAME(name) __closesthit__##name
#define RG_NAME_STR(name) AMBER_STRINGIFY(RG_NAME(name)) 
#define MISS_NAME_STR(name) AMBER_STRINGIFY(MISS_NAME(name)) 
#define CH_NAME_STR(name) AMBER_STRINGIFY(CH_NAME(name)) 

namespace amber
{
	struct RayGenData
	{
	};
	struct MissData
	{
		float3 bg_color;
	};
	struct HitGroupData
	{
	};

	struct MaterialGPU
	{
		float3 base_color = make_float3(0.9f, 0.9f, 0.9f);
		float metallic = 0.0f;

		float specular = 0.0f;
		float roughness = 1.0f;
		float specular_tint = 0.0f;
		float anisotropy = 0.0f;

		float sheen = 0.0f;
		float sheen_tint = 0.0f;
		float clearcoat = 0.0f;
		float clearcoat_gloss = 0.0f;

		float ior = 1.5f;
		float specular_transmission = 0.0f;

		int diffuse_tex_id;
	};
	struct MeshGPU
	{
		unsigned int positions_offset;
		unsigned int positions_count;
		unsigned int uvs_offset;
		unsigned int uvs_count;
		unsigned int normals_offset;
		unsigned int normals_count;
		unsigned int indices_offset;
		unsigned int indices_count;
		unsigned int material_idx;
	};

	struct Params
	{
		OptixTraversableHandle	handle;
		uchar4*					image;
		unsigned int			sample_count;
		unsigned int			frame_index;

		float3					cam_eye;
		float3					cam_u, cam_v, cam_w;
		float					cam_fovy;
		float					cam_aspect_ratio;
#ifdef __CUDACC__
		float3*					vertices;
		float3*					normals;
		float2*					uvs;
		uint3*					indices;
		cudaTextureObject_t*	textures;
		MaterialGPU*			materials;
		MeshGPU*				meshes;
#else
		CUdeviceptr				vertices;
		CUdeviceptr				normals;
		CUdeviceptr				uvs;
		CUdeviceptr				indices;
		CUdeviceptr				textures;
		CUdeviceptr				materials;
		CUdeviceptr				meshes;
#endif
		cudaTextureObject_t		sky;
	};

	struct Payload
	{
		float3		 radiance;
		float3		 emissive;
		float3		 origin;
		float3		 direction;
		int			 done;

		float3       attenuation;
		unsigned int seed;
		int          depth;
	};
}

