#pragma once

namespace lavender
{
	enum class LightType
	{
		Directional,
		Point
	};

	struct Light 
	{
		LightType type;
		Vector3 color;

		struct Directional
		{
			Vector3 direction;
		};
		struct Point
		{
			Vector3 position;
		};
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
			l.directional = Light::Directional(std::forward<Args>(args)...);
		}
		else if constexpr (type == LightType::Point)
		{
			l.point = Light::Point(std::forward<Args>(args)...);
		}
		return l;
	}
}