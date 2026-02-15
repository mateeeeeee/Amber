#include "Sampler.h"
#include <cmath>
#include <algorithm>
#include <type_traits>

namespace amber
{
	static Vector4 DecodeTexel(Texture const& texture, Int32 x, Int32 y)
	{
		x = std::clamp(x, 0, (Int32)texture.width  - 1);
		y = std::clamp(y, 0, (Int32)texture.height - 1);

		Uint32 const channels = GetChannelCount(texture.format);
		Bool   const srgb     = IsSRGB(texture.format);
		Bool   const is_float = IsFloat(texture.format);

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

	static Vector4u DecodeTexelUInt(Texture const& texture, Int32 x, Int32 y)
	{
		x = std::clamp(x, 0, (Int32)texture.width  - 1);
		y = std::clamp(y, 0, (Int32)texture.height - 1);

		Uint32 const channels = GetChannelCount(texture.format);
		Vector4u result(0, 0, 0, 1);
		Uint64 const texel_offset = ((Uint64)y * texture.width + x) * channels;
		switch (texture.format)
		{
		case TextureFormat::R8U:
		case TextureFormat::RG8U:
		case TextureFormat::RGBA8U:
		{
			Uint8 const* p = static_cast<Uint8 const*>(texture.data) + texel_offset;
			if (channels > 0) result.x = p[0];
			if (channels > 1) result.y = p[1];
			if (channels > 2) result.z = p[2];
			if (channels > 3) result.w = p[3];
			break;
		}
		case TextureFormat::R16U:
		{
			Uint16 const* p = static_cast<Uint16 const*>(texture.data) + texel_offset;
			result.x = p[0];
			break;
		}
		case TextureFormat::R32U:
		{
			Uint32 const* p = static_cast<Uint32 const*>(texture.data) + texel_offset;
			result.x = p[0];
			break;
		}
		default: break;
		}

		return result;
	}

	static Vector4i DecodeTexelInt(Texture const& texture, Int32 x, Int32 y)
	{
		x = std::clamp(x, 0, (Int32)texture.width  - 1);
		y = std::clamp(y, 0, (Int32)texture.height - 1);

		Uint64 const texel_offset = ((Uint64)y * texture.width + x);
		Vector4i result(0, 0, 0, 1);
		switch (texture.format)
		{
		case TextureFormat::R32I:
		{
			Int32 const* p = static_cast<Int32 const*>(texture.data) + texel_offset;
			result.x = p[0];
			break;
		}
		default: break;
		}
		return result;
	}

	template<FilterMode Filter, WrapMode Wrap>
	static Vector2 ApplyWrap(Vector2 uv)
	{
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
		return uv;
	}

	template<FilterMode Filter, WrapMode Wrap>
	Vector4 Sampler<Filter, Wrap>::SampleRaw(Texture const& texture, Vector2 uv) const
	{
		Int32 w = static_cast<Int32>(texture.width);
		Int32 h = static_cast<Int32>(texture.height);

		uv = ApplyWrap<Filter, Wrap>(uv);
		if constexpr (Filter == FilterMode::Nearest)
		{
			return DecodeTexel(texture, static_cast<Int32>(uv.x * w), static_cast<Int32>(uv.y * h));
		}
		else 
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

	template<FilterMode Filter, WrapMode Wrap>
	Vector4u Sampler<Filter, Wrap>::SampleRawUInt(Texture const& texture, Vector2 uv) const
	{
		Int32 w = static_cast<Int32>(texture.width);
		Int32 h = static_cast<Int32>(texture.height);

		uv = ApplyWrap<Filter, Wrap>(uv);
		return DecodeTexelUInt(texture, static_cast<Int32>(uv.x * w), static_cast<Int32>(uv.y * h));
	}

	template<FilterMode Filter, WrapMode Wrap>
	Vector4i Sampler<Filter, Wrap>::SampleRawInt(Texture const& texture, Vector2 uv) const
	{
		Int32 w = static_cast<Int32>(texture.width);
		Int32 h = static_cast<Int32>(texture.height);

		uv = ApplyWrap<Filter, Wrap>(uv);
		return DecodeTexelInt(texture, static_cast<Int32>(uv.x * w), static_cast<Int32>(uv.y * h));
	}

	template<FilterMode Filter, WrapMode Wrap>
	template<typename T>
	T Sampler<Filter, Wrap>::Sample(Texture const& texture, Vector2 uv) const
	{
		if constexpr (std::is_same_v<T, Uint32>)
		{
			return SampleRawUInt(texture, uv).x;
		}
		else if constexpr (std::is_same_v<T, Vector2u>)
		{
			Vector4u v = SampleRawUInt(texture, uv);
			return Vector2u(v.x, v.y);
		}
		else if constexpr (std::is_same_v<T, Vector3u>)
		{
			Vector4u v = SampleRawUInt(texture, uv);
			return Vector3u(v.x, v.y, v.z);
		}
		else if constexpr (std::is_same_v<T, Vector4u>)
		{
			return SampleRawUInt(texture, uv);
		}
		else if constexpr (std::is_same_v<T, Int32>)
		{
			return SampleRawInt(texture, uv).x;
		}
		else if constexpr (std::is_same_v<T, Vector2i>)
		{
			Vector4i v = SampleRawInt(texture, uv);
			return Vector2i(v.x, v.y);
		}
		else if constexpr (std::is_same_v<T, Vector3i>)
		{
			Vector4i v = SampleRawInt(texture, uv);
			return Vector3i(v.x, v.y, v.z);
		}
		else if constexpr (std::is_same_v<T, Vector4i>)
		{
			return SampleRawInt(texture, uv);
		}
		else
		{
			Vector4 v = SampleRaw(texture, uv);
			if constexpr (std::is_same_v<T, Float>)   return v.x;
			if constexpr (std::is_same_v<T, Vector2>) return Vector2(v.x, v.y);
			if constexpr (std::is_same_v<T, Vector3>) return Vector3(v.x, v.y, v.z);
			if constexpr (std::is_same_v<T, Vector4>) return v;
		}
	}

	template struct Sampler<FilterMode::Bilinear,  WrapMode::Repeat>;
	template struct Sampler<FilterMode::Bilinear,  WrapMode::Clamp>;
	template struct Sampler<FilterMode::Nearest,   WrapMode::Repeat>;
	template struct Sampler<FilterMode::Nearest,   WrapMode::Clamp>;
	template struct Sampler<FilterMode::Trilinear, WrapMode::Repeat>;
	template struct Sampler<FilterMode::Trilinear, WrapMode::Clamp>;

#define INSTANTIATE_SAMPLE(Filter, Wrap) \
	template Float    Sampler<Filter, Wrap>::Sample<Float>   (Texture const&, Vector2) const; \
	template Vector2  Sampler<Filter, Wrap>::Sample<Vector2> (Texture const&, Vector2) const; \
	template Vector3  Sampler<Filter, Wrap>::Sample<Vector3> (Texture const&, Vector2) const; \
	template Vector4  Sampler<Filter, Wrap>::Sample<Vector4> (Texture const&, Vector2) const; \
	template Uint32   Sampler<Filter, Wrap>::Sample<Uint32>  (Texture const&, Vector2) const; \
	template Vector2u Sampler<Filter, Wrap>::Sample<Vector2u>(Texture const&, Vector2) const; \
	template Vector3u Sampler<Filter, Wrap>::Sample<Vector3u>(Texture const&, Vector2) const; \
	template Vector4u Sampler<Filter, Wrap>::Sample<Vector4u>(Texture const&, Vector2) const; \
	template Int32    Sampler<Filter, Wrap>::Sample<Int32>   (Texture const&, Vector2) const; \
	template Vector2i Sampler<Filter, Wrap>::Sample<Vector2i>(Texture const&, Vector2) const; \
	template Vector3i Sampler<Filter, Wrap>::Sample<Vector3i>(Texture const&, Vector2) const; \
	template Vector4i Sampler<Filter, Wrap>::Sample<Vector4i>(Texture const&, Vector2) const;

	INSTANTIATE_SAMPLE(FilterMode::Bilinear,  WrapMode::Repeat)
	INSTANTIATE_SAMPLE(FilterMode::Bilinear,  WrapMode::Clamp)
	INSTANTIATE_SAMPLE(FilterMode::Nearest,   WrapMode::Repeat)
	INSTANTIATE_SAMPLE(FilterMode::Nearest,   WrapMode::Clamp)
	INSTANTIATE_SAMPLE(FilterMode::Trilinear, WrapMode::Repeat)
	INSTANTIATE_SAMPLE(FilterMode::Trilinear, WrapMode::Clamp)

#undef INSTANTIATE_SAMPLE
}
