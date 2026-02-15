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
		T Sample(Texture const& texture, Vector2 uv) const;

	private:
		Vector4  SampleRaw(Texture const& texture, Vector2 uv) const;
		Vector4u SampleRawUInt(Texture const& texture, Vector2 uv) const;
		Vector4i SampleRawInt(Texture const& texture, Vector2 uv) const;
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
