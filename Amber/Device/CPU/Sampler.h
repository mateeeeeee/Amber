#pragma once
#include "Texture.h"

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

	template<FilterMode Filter, WrapMode Wrap>
	struct Sampler
	{
		template<typename T>
		T Sample(Texture const& texture, Vector2 uv) const
		{
			Vector4 v = SampleRaw(texture, uv);
			if constexpr (std::is_same_v<T, Float>)   return v.x;
			if constexpr (std::is_same_v<T, Vector2>) return Vector2(v.x, v.y);
			if constexpr (std::is_same_v<T, Vector3>) return Vector3(v.x, v.y, v.z);
			if constexpr (std::is_same_v<T, Vector4>) return v;
		}

	private:
		Vector4 SampleRaw(Texture const& texture, Vector2 uv) const;
	};

	using SamplerBilinearRepeat  = Sampler<FilterMode::Bilinear,  WrapMode::Repeat>;
	using SamplerBilinearClamp   = Sampler<FilterMode::Bilinear,  WrapMode::Clamp>;
	using SamplerNearestRepeat   = Sampler<FilterMode::Nearest,   WrapMode::Repeat>;
	using SamplerNearestClamp    = Sampler<FilterMode::Nearest,   WrapMode::Clamp>;
	using SamplerTrilinearRepeat = Sampler<FilterMode::Trilinear, WrapMode::Repeat>;
	using SamplerTrilinearClamp  = Sampler<FilterMode::Trilinear, WrapMode::Clamp>;

	inline constexpr SamplerBilinearRepeat  BilinearRepeat{};
	inline constexpr SamplerBilinearClamp   BilinearClamp{};
	inline constexpr SamplerNearestRepeat   NearestRepeat{};
	inline constexpr SamplerNearestClamp    NearestClamp{};
	inline constexpr SamplerTrilinearRepeat TrilinearRepeat{};
	inline constexpr SamplerTrilinearClamp  TrilinearClamp{};
}
