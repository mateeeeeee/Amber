#pragma once

namespace amber
{

	enum class LightType : uint32
	{
		Directional,
		Point,
		Spot,
		Area,
		Environmental,
	};

	struct Light
	{
		LightType	type;
		Vector3		direction;
		Vector3		position;
		Vector3		color;
	};
}