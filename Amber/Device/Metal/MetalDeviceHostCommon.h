#ifndef _METALDEVICEHOSTCOMMON_
#define _METALDEVICEHOSTCOMMON_

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

using Float = float;
using Float2 = float2;
using Float3 = float3;
using Float4 = float4;
using Int = int;
using Uint = unsigned int;
using Uint32 = uint;
using Bool32 = bool;

#else
#include "Math/MathTypes.h"
#include <Metal/Metal.h>

using Float = float;
using Float2 = amber::Vector2;
using Float3 = amber::Vector3;
using Float4 = amber::Vector4;
using Int = int;
using Uint = unsigned int;
using Uint32 = uint32_t;
using Bool32 = bool;

#endif

namespace amber
{
	struct alignas(16) MaterialGPU
	{
		Float4 base_color;
		Int    diffuse_tex_id;
		Int    normal_tex_id;
		Int    padding0;
		Int    padding1;

		Float4 emissive_color; 
		Int    emissive_tex_id;
		Int    metallic_roughness_tex_id;
		Float  metallic;

		Float  specular;
		Float  roughness;
		Float  specular_tint;
		Float  anisotropy;

		Float  alpha_cutoff;
		Float  sheen;
		Float  sheen_tint;
		Float  clearcoat;

		Float  clearcoat_gloss;
		Float  ior;
		Float  specular_transmission;
		Float  padding2;
	};

	struct alignas(16) MeshGPU
	{
		Uint32 positions_offset;
		Uint32 positions_count;
		Uint32 uvs_offset;
		Uint32 uvs_count;
		Uint32 normals_offset;
		Uint32 normals_count;
		Uint32 indices_offset;
		Uint32 triangle_count;
		Uint32 material_idx;
		Uint32 padding[3];
	};

	enum LightGPUType : Int
	{
		LightGPUType_Directional = 0,
		LightGPUType_Point = 1,
		LightGPUType_Spot = 2,
		LightGPUType_Area = 3,
		LightGPUType_Environmental = 4
	};

	struct alignas(16) LightGPU
	{
		Uint32		type;
		Float		padding0;
		Float		padding1;
		Float		padding2;
		Float4		direction;  
		Float4		position;   
		Float4		color;      
	};

	struct RenderParams
	{
		Float4		cam_eye;
		Float4		cam_u;
		Float4		cam_v;
		Float4		cam_w;

		Float		cam_fovy;
		Float		cam_aspect_ratio;
		Uint32		sample_count;
		Uint32		frame_index;

		Uint32		max_depth;
		Uint32		output_type;
		Uint32		light_count;
		Uint32		width;

		Uint32		height;
		Uint32		padding1;
		Uint32		padding2;
		Uint32		padding3;
	};

#ifdef __METAL_VERSION__
#define CONSTANT_PTR(x) constant x*
#else
#define CONSTANT_PTR(x) Uint64
#endif

#define MAX_TEXTURES 256

	struct SceneResources
	{
		CONSTANT_PTR(float3) vertices;
		CONSTANT_PTR(float3) normals;
		CONSTANT_PTR(float2) uvs;
		CONSTANT_PTR(Uint) indices;
		CONSTANT_PTR(MeshGPU) meshes;
		CONSTANT_PTR(MaterialGPU) materials;
		CONSTANT_PTR(LightGPU) lights;
#ifdef __METAL_VERSION__
		array<texture2d<float>, MAX_TEXTURES> textures;
#else
		MTLResourceID textures[MAX_TEXTURES];
#endif
	};
}

#endif