#pragma once
#include <vector>

namespace amber
{
	enum class ImageFormat : Uint8
	{
		PNG,
		JPG,
		HDR,
		TGA,
		BMP
	};
	void WriteImageToFile(ImageFormat type, Char const* filename, Uint32 width, Uint32 height, void const* data, Uint32 stride);

	class Image
	{
	public:
		explicit Image(Char const* file, Bool srgb = false);

		Int32 GetWidth() const { return width; }
		Int32 GetHeight() const { return height; }

		template<typename T = Uint8>
		T const* GetData() const
		{
			if constexpr(std::is_same_v<T, Float>)
			{
				AMBER_ASSERT(hdr);
				return hdr_data.data();
			}
			else
			{
				return reinterpret_cast<T const*>(data.data());
			}
		}
		Bool IsSRGB() const { return srgb; }
		Bool IsHDR() const { return hdr; }

	private:
		Int32 width;
		Int32 height;
		Int32 channels;
		std::vector<Uint8> data;
		std::vector<Float> hdr_data;
		Bool srgb;
		Bool hdr;
	};

}