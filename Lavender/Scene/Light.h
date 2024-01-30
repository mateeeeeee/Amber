#pragma once

namespace lavender
{
	enum class LightType : uint8
	{
		Directional,
		Point
	};

	struct Directional
	{
		Vector3 direction;
	};

	struct Point
	{
		Vector3 position;
	};

	struct Light
	{
		LightType type;
		Vector3 color;
		union
		{
			Directional directional;
			Point point;
		};
	};

	template<LightType type, typename... Args>
	inline Light MakeLight(Args&&... args)
	{
		Light l{ type };
		if constexpr (type == LightType::Directional)
		{
			l.directional = Directional(std::forward<Args>(args)...);
		}
		else if constexpr (type == LightType::Point)
		{
			l.point = Point(std::forward<Args>(args)...);
		}
		return l;
	}
}