#pragma once
#include "Cuda/CudaCore.h"

namespace lavender
{
	struct Camera
	{
		Vector3		position;
		Quaternion	rotation;
		float fov;

		float lens_radius;
		float focal_distance;

		float shutter_start;
		float shutter_end;
	};
}