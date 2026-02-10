#pragma once
#include "Utilities/ImageUtil.h"

namespace amber
{
	enum class FilterMode : Uint8
	{
		Nearest,
		Bilinear,
		Trilinear,
		Anisotropic
	};

	enum class WrapMode : Uint8
	{
		Repeat,
		Clamp,
		Mirror,
		Border
	};

	struct Sampler
	{
		FilterMode filter = FilterMode::Bilinear;
		WrapMode   wrap   = WrapMode::Repeat;

		Vector4 Sample(Image const& image, Vector2 uv) const;
	};
}
