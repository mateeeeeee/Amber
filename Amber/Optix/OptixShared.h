#pragma once
#include <vector_types.h>
#include <optix_types.h>

#define RG_NAME(name) __raygen__##name
#define MISS_NAME(name) __miss__##name
#define CH_NAME(name) __closesthit__##name
#define AH_NAME(name) __anyhit__##name
#define RG_NAME_STR(name)	AMBER_STRINGIFY(RG_NAME(name)) 
#define MISS_NAME_STR(name) AMBER_STRINGIFY(MISS_NAME(name)) 
#define CH_NAME_STR(name)	AMBER_STRINGIFY(CH_NAME(name)) 
#define AH_NAME_STR(name)	AMBER_STRINGIFY(AH_NAME(name)) 

namespace amber
{
	struct MaterialGPU
	{
		float3 base_color;
		Sint32 diffuse_tex_id = -1;
		Sint32 normal_tex_id = -1;

		float3 emissive_color;
		Sint32 emissive_tex_id = -1;

		Sint32 metallic_roughness_tex_id = -1;
		float metallic = 0.0f;
		float specular = 0.0f;
		float roughness = 1.0f;
		float specular_tint = 0.0f;
		float anisotropy = 0.0f;
		float alpha_cutoff = 0.5f;

		float sheen = 0.0f;
		float sheen_tint = 0.0f;
		float clearcoat = 0.0f;
		float clearcoat_gloss = 0.0f;

		float ior = 1.5f;
		float specular_transmission = 0.0f;

	};
	struct MeshGPU
	{
		Uint32 positions_offset;
		Uint32 positions_count;
		Uint32 uvs_offset;
		Uint32 uvs_count;
		Uint32 normals_offset;
		Uint32 normals_count;
		Uint32 indices_offset;
		Uint32 indices_count;
		Uint32 material_idx;
	};

	#define LightType_Directional 0
	#define LightType_Point 1
	#define LightType_Spot 2
	#define LightType_Area 3
	#define LightType_Environmental 4

	struct LightGPU
	{
		Uint32		type;
		float3		direction;
		float3		position;
		float3		color;
	};

	struct LaunchParams
	{
		OptixTraversableHandle	traversable;
		uchar4*					output;
		float4*					accum;
		Uint32			        sample_count;
		Uint32			        frame_index;
		Uint32			        max_depth;

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
		LightGPU*				lights;
#else
		CUdeviceptr				vertices;
		CUdeviceptr				normals;
		CUdeviceptr				uvs;
		CUdeviceptr				indices;
		CUdeviceptr				textures;
		CUdeviceptr				materials;
		CUdeviceptr				meshes;
		CUdeviceptr				lights;
#endif
		Uint32					light_count;
		cudaTextureObject_t		sky;
	};

	struct HitRecord
	{
		Uint32		 depth;
		Bool32		 hit;
		float3		 P;
		float3		 N;
		float2       uv;
		Uint32       material_idx;
	};
}

