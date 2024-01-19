#pragma once
#include "Cuda/CudaCore.h"

namespace lavender
{
	enum class MaterialType
	{
		Lambertian,
		Mirror
	};

	struct Material
	{
		MaterialType type;
		union
		{
			struct Lambertian
			{
				Vector3 albedo;
			} lambertian;

			struct Mirror
			{
				Vector3 albedo;
				float fuzz;
			} mirror;
		};
	};

	template<MaterialType type, typename... Args>
	inline Material MakeMaterial(Args&&... args)
	{
		Material m{ type };
		if constexpr (type == MaterialType::Lambertian)
		{
			m.lambertian = Lambertian(std::forward<Args>(args)...);
		}
		else if constexpr (type == MaterialType::Mirror)
		{
			m.mirror = Mirror(std::forward<Args>(args)...);
		}
		return m;
	}
}