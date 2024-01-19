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
		union 
		{
			struct Directional
			{
				Vector3 direction;
			} directional;

			struct Point
			{
				Vector3 position;
				float   range;
			} point;
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