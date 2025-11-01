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
			return reinterpret_cast<T const*>(data.data());
		}
		Bool IsSRGB() const { return srgb; }

	private:
		Int32 width;
		Int32 height;
		Int32 channels;
		std::vector<Uint8> data;
		Bool srgb;
	};

}