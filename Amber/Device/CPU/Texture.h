#pragma once

namespace amber
{
	enum class TextureFormat : Uint8
	{
		R8,
		RG8,
		RGB8,
		RGBA8,

		R8_SRGB,
		RG8_SRGB,
		RGB8_SRGB,
		RGBA8_SRGB,

		R16F,
		RG16F,
		RGB16F,
		RGBA16F,

		R32F,
		RG32F,
		RGB32F,
		RGBA32F,

		R8U,
		RG8U,
		RGBA8U,
		R16U,
		R32U,

		R32I,

		Unknown
	};

	inline Uint32 GetChannelCount(TextureFormat format)
	{
		switch (format)
		{
		case TextureFormat::R8:
		case TextureFormat::R8_SRGB:
		case TextureFormat::R16F:
		case TextureFormat::R32F:
		case TextureFormat::R8U:
		case TextureFormat::R16U:
		case TextureFormat::R32U:
		case TextureFormat::R32I:
			return 1;
		case TextureFormat::RG8:
		case TextureFormat::RG8_SRGB:
		case TextureFormat::RG16F:
		case TextureFormat::RG32F:
		case TextureFormat::RG8U:
			return 2;
		case TextureFormat::RGB8:
		case TextureFormat::RGB8_SRGB:
		case TextureFormat::RGB16F:
		case TextureFormat::RGB32F:
			return 3;
		case TextureFormat::RGBA8:
		case TextureFormat::RGBA8_SRGB:
		case TextureFormat::RGBA16F:
		case TextureFormat::RGBA32F:
		case TextureFormat::RGBA8U:
			return 4;
		default:
			return 0;
		}
	}

	inline Bool IsFloat(TextureFormat format)
	{
		switch (format)
		{
		case TextureFormat::R16F:
		case TextureFormat::RG16F:
		case TextureFormat::RGB16F:
		case TextureFormat::RGBA16F:
		case TextureFormat::R32F:
		case TextureFormat::RG32F:
		case TextureFormat::RGB32F:
		case TextureFormat::RGBA32F:
			return true;
		default:
			return false;
		}
	}

	inline Bool IsSRGB(TextureFormat format)
	{
		switch (format)
		{
		case TextureFormat::R8_SRGB:
		case TextureFormat::RG8_SRGB:
		case TextureFormat::RGB8_SRGB:
		case TextureFormat::RGBA8_SRGB:
			return true;
		default:
			return false;
		}
	}

	inline Bool IsUInt(TextureFormat format)
	{
		switch (format)
		{
		case TextureFormat::R8U:
		case TextureFormat::RG8U:
		case TextureFormat::RGBA8U:
		case TextureFormat::R16U:
		case TextureFormat::R32U:
			return true;
		default:
			return false;
		}
	}

	inline Bool IsInt(TextureFormat format)
	{
		switch (format)
		{
		case TextureFormat::R32I:
			return true;
		default:
			return false;
		}
	}

	struct Texture
	{
		void const*   data   = nullptr;
		Uint32        width  = 0;
		Uint32        height = 0;
		TextureFormat format = TextureFormat::Unknown;
	};
}
