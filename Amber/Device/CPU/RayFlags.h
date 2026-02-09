#pragma once
#include "Utilities/EnumUtil.h"

namespace amber
{
	enum class RayGeometryFlags : Uint8
	{
		None              = 0,
		Opaque            = 1 << 0,
		NoDuplicateAnyHit = 1 << 1,
	};
	ENABLE_ENUM_BIT_OPERATORS(RayGeometryFlags);
}
