#pragma once
#include <vector_types.h>
#include <optix_types.h>

#define RG_NAME(name) __raygen__##name
#define MISS_NAME(name) __miss__##name
#define CH_NAME(name) __closesthit__##name
#define RG_NAME_STR(name) LAV_STRINGIFY(RG_NAME(name)) 
#define MISS_NAME_STR(name) LAV_STRINGIFY(MISS_NAME(name)) 
#define CH_NAME_STR(name) LAV_STRINGIFY(CH_NAME(name)) 

namespace lavender
{
	struct Params
	{
		OptixTraversableHandle handle;
		uchar4*				   image;
		unsigned int		   sample_count;
		unsigned int		   frame_index;
		float3                 cam_eye;
		float3                 cam_u, cam_v, cam_w;
		float				   cam_fovy;
		float				   cam_aspect_ratio;
	};


	struct RayGenData
	{
		// No data needed
	};


	struct MissData
	{
		float3 bg_color;
	};


	struct HitGroupData
	{
		// No data needed
	};

}

