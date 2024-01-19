#pragma once
#include "Material.h"
#include "Cuda/CudaAlloc.h"

namespace lavender
{
	struct Sphere
	{
		Sphere() {}
	};

	enum class GeometryType
	{
		Sphere
	};

	struct Primitive
	{
		GeometryType type;
		union
		{
			Sphere sphere;
		};

		LAV_HOST_DEVICE bool Intersect()
		{
			switch (type)
			{
			case GeometryType::Sphere:
			default:
				return false;
			}
		}
	};

	template<GeometryType type, typename... Args>
	inline Primitive MakePrimitive(Args&&... args)
	{
		Primitive p{ type };
		if constexpr (type == GeometryType::Sphere)
		{
			p.sphere = Sphere(std::forward<Args>(args)...);
		}
		return p;
	}

	struct PrimitiveContainer
	{
		TypedCudaAlloc<Primitive> primitives;
		uint64 primitive_count;

		LAV_HOST_DEVICE bool Intersect() const
		{
			
		}
	};
}