#include "Sampler.h"
#include <cmath>
#include <algorithm>

namespace amber
{
	static Vector4 DecodeTexel(Texture const& texture, Int32 x, Int32 y)
	{
		x = std::clamp(x, 0, (Int32)texture.width  - 1);
		y = std::clamp(y, 0, (Int32)texture.height - 1);

		Uint32     const channels = GetChannelCount(texture.format);
		Bool       const srgb     = IsSRGB(texture.format);
		Bool       const is_float = IsFloat(texture.format);

		Vector4 result(0.0f, 0.0f, 0.0f, 1.0f);

		if (is_float)
		{
			Float const* p = static_cast<Float const*>(texture.data) + (y * texture.width + x) * channels;
			if (channels > 0) result.x = p[0];
			if (channels > 1) result.y = p[1];
			if (channels > 2) result.z = p[2];
			if (channels > 3) result.w = p[3];
		}
		else
		{
			Uint8 const* p = static_cast<Uint8 const*>(texture.data) + (y * texture.width + x) * channels;
			if (channels > 0) result.x = p[0] / 255.0f;
			if (channels > 1) result.y = p[1] / 255.0f;
			if (channels > 2) result.z = p[2] / 255.0f;
			if (channels > 3) result.w = p[3] / 255.0f;

			if (srgb)
			{
				result.x = std::pow(result.x, 2.2f);
				result.y = std::pow(result.y, 2.2f);
				result.z = std::pow(result.z, 2.2f);
			}
		}

		return result;
	}

	template<FilterMode Filter, WrapMode Wrap>
	Vector4 Sampler<Filter, Wrap>::SampleRaw(Texture const& texture, Vector2 uv) const
	{
		Int32 w = static_cast<Int32>(texture.width);
		Int32 h = static_cast<Int32>(texture.height);

		if constexpr (Wrap == WrapMode::Repeat)
		{
			uv.x = uv.x - std::floor(uv.x);
			uv.y = 1.0f - (uv.y - std::floor(uv.y));
		}
		else if constexpr (Wrap == WrapMode::Clamp)
		{
			uv.x = std::clamp(uv.x, 0.0f, 1.0f);
			uv.y = 1.0f - std::clamp(uv.y, 0.0f, 1.0f);
		}

		if constexpr (Filter == FilterMode::Nearest)
		{
			return DecodeTexel(texture, static_cast<Int32>(uv.x * w), static_cast<Int32>(uv.y * h));
		}
		else // Bilinear, fallback for Trilinear/Anisotropic
		{
			Float fx = uv.x * w - 0.5f;
			Float fy = uv.y * h - 0.5f;
			Int32 x0 = static_cast<Int32>(std::floor(fx));
			Int32 y0 = static_cast<Int32>(std::floor(fy));
			Float tx = fx - std::floor(fx);
			Float ty = fy - std::floor(fy);

			Int32 x1, y1;
			if constexpr (Wrap == WrapMode::Repeat)
			{
				x1 = ((x0 + 1) % w + w) % w;
				y1 = ((y0 + 1) % h + h) % h;
				x0 = ((x0     % w) + w) % w;
				y0 = ((y0     % h) + h) % h;
			}
			else
			{
				x0 = std::clamp(x0,     0, w - 1);
				x1 = std::clamp(x0 + 1, 0, w - 1);
				y0 = std::clamp(y0,     0, h - 1);
				y1 = std::clamp(y0 + 1, 0, h - 1);
			}

			Vector4 c00 = DecodeTexel(texture, x0, y0);
			Vector4 c10 = DecodeTexel(texture, x1, y0);
			Vector4 c01 = DecodeTexel(texture, x0, y1);
			Vector4 c11 = DecodeTexel(texture, x1, y1);

			Vector4 c0 = c00 * (1.0f - tx) + c10 * tx;
			Vector4 c1 = c01 * (1.0f - tx) + c11 * tx;
			return c0 * (1.0f - ty) + c1 * ty;
		}
	}

	template struct Sampler<FilterMode::Bilinear,  WrapMode::Repeat>;
	template struct Sampler<FilterMode::Bilinear,  WrapMode::Clamp>;
	template struct Sampler<FilterMode::Nearest,   WrapMode::Repeat>;
	template struct Sampler<FilterMode::Nearest,   WrapMode::Clamp>;
	template struct Sampler<FilterMode::Trilinear, WrapMode::Repeat>;
	template struct Sampler<FilterMode::Trilinear, WrapMode::Clamp>;
}
