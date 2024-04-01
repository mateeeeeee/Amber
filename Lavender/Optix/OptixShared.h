#pragma once
#include <vector_types.h>
#include <optix_types.h>

namespace lavender
{
	struct HitData
	{
		float3 color;
	};

	struct RaygenData
	{

	};

	struct MissData
	{
		float3 color;
	};

	struct LaunchParams
	{
		uint8*				   image;
		uint32		           image_width;
		uint32		           image_height;
		OptixTraversableHandle handle;
	};
}

