#pragma once
#include <optix_types.h>
#include "Kernels/Impl/MathDefines.cuh"

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
		Float3 base_color;
		Int    diffuse_tex_id = -1;
		Int    normal_tex_id = -1;

		Float3 emissive_color;
		Int    emissive_tex_id = -1;

		Int    metallic_roughness_tex_id = -1;
		Float  metallic = 0.0f;
		Float  specular = 0.0f;
		Float  roughness = 1.0f;
		Float  specular_tint = 0.0f;
		Float  anisotropy = 0.0f;
		Float  alpha_cutoff = 0.5f;

		Float  sheen = 0.0f;
		Float  sheen_tint = 0.0f;
		Float  clearcoat = 0.0f;
		Float  clearcoat_gloss = 0.0f;

		Float  ior = 1.5f;
		Float  specular_transmission = 0.0f;
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

	enum LightGPUType : Int
	{
		LightGPUType_Directional,
		LightGPUType_Point,
		LightGPUType_Spot,
		LightGPUType_Area,
		LightGPUType_Environmental
	};

	enum PathTracerOutputGPU : Int
	{
		PathTracerOutputGPU_Final,
		PathTracerOutputGPU_Albedo,
		PathTracerOutputGPU_Normal,
		PathTracerOutputGPU_UV,
		PathTracerOutputGPU_MaterialID,
		PathTracerOutputGPU_Custom
	};

	struct LightGPU
	{
		Uint32		type;
		Float3		direction;
		Float3		position;
		Float3		color;
	};

	struct LaunchParams
	{
		OptixTraversableHandle	traversable;
		Float3* accum_buffer;
		Float3* debug_buffer;
		Uint32			        sample_count;
		Uint32			        frame_index;
		Uint32			        max_depth;
		Uint32					output_type;

		Float3					cam_eye;
		Float3					cam_u, cam_v, cam_w;
		Float					cam_fovy;
		Float					cam_aspect_ratio;
#ifdef __CUDACC__
		Float3*					vertices;
		Float3*					normals;
		Float2*					uvs;
		Uint3*					indices;
		cudaTextureObject_t*	textures;
		MaterialGPU*			materials;
		MeshGPU*				meshes;
		LightGPU*				lights;
		float3*					denoiser_albedo;
		float3*					denoiser_normals;
		cudaTextureObject_t		sky;
#else
		CUdeviceptr				vertices;
		CUdeviceptr				normals;
		CUdeviceptr				uvs;
		CUdeviceptr				indices;
		CUdeviceptr				textures;
		CUdeviceptr				materials;
		CUdeviceptr				meshes;
		CUdeviceptr				lights;
		CUdeviceptr				denoiser_albedo;
		CUdeviceptr				denoiser_normals;
		CUdeviceptr				sky;
#endif
		Uint32					light_count;
	};

	struct HitRecord
	{
		Uint32		 depth;
		Bool32		 hit;
		Float3		 P;
		Float3		 N;
		Float2       uv;
		Uint32       material_idx;
	};
}

