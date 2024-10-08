#pragma once
#include "MathTypes.h"

namespace amber
{
	inline Vector3 ConvertElevationAndAzimuthToDirection(float elevation, float azimuth)
	{
		float phi = DirectX::XMConvertToRadians(azimuth);
		float theta = DirectX::XMConvertToRadians(elevation);
		float x = cos(theta) * sin(phi);
		float y = sin(theta);
		float z = cos(theta) * cos(phi);
		return Vector3(x, y, z);
	}
	inline void ConvertDirectionToAzimuthAndElevation(Vector3 const& direction, float& elevation, float& azimuth)
	{
		azimuth = atan2(direction.x, direction.z);
		if (azimuth < 0)  azimuth += DirectX::XM_2PI;
		elevation = asin(direction.y);

		azimuth = DirectX::XMConvertToDegrees(azimuth);
		elevation = DirectX::XMConvertToDegrees(elevation);
	}

	template<typename T>
	auto Clamp(T v, T min, T max)
	{
		return v < min ? min : (v > max ? max : v);
	}

}