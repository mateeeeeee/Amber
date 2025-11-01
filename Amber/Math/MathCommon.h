#pragma once
#include "MathTypes.h"
#include <cmath>

namespace amber
{
	constexpr Float PI = 3.14159265358979323846f;
	constexpr Float TWO_PI = 2.0f * PI;

	inline Float DegreesToRadians(Float degrees)
	{
		return degrees * (PI / 180.0f);
	}

	inline Float RadiansToDegrees(Float radians)
	{
		return radians * (180.0f / PI);
	}

	inline Vector3 ConvertElevationAndAzimuthToDirection(Float elevation, Float azimuth)
	{
		Float phi = DegreesToRadians(azimuth);
		Float theta = DegreesToRadians(elevation);
		Float x = std::cos(theta) * std::sin(phi);
		Float y = std::sin(theta);
		Float z = std::cos(theta) * std::cos(phi);
		return Vector3(x, y, z);
	}

	inline void ConvertDirectionToAzimuthAndElevation(Vector3 const& direction, Float& elevation, Float& azimuth)
	{
		azimuth = std::atan2(direction.x, direction.z);
		if (azimuth < 0)  azimuth += TWO_PI;
		elevation = std::asin(direction.y);

		azimuth = RadiansToDegrees(azimuth);
		elevation = RadiansToDegrees(elevation);
	}

	template<typename T>
	auto Clamp(T v, T min, T max)
	{
		return v < min ? min : (v > max ? max : v);
	}

}