#pragma once

namespace amber
{
	enum class ImageFormat : uint8
	{
		PNG,
		JPG,
		HDR,
		TGA,
		BMP
	};
	void WriteImageToFile(ImageFormat type, char const* filename, uint32 width, uint32 height, void const* data, uint32 stride);

	class Image
	{
	public:
		explicit Image(char const* file, bool srgb = false);

		int32 GetWidth() const { return width; }
		int32 GetHeight() const { return height; }

		template<typename T = uint8>
		T const* GetData() const
		{
			return reinterpret_cast<T const*>(data.data());
		}
		bool IsSRGB() const { return srgb; }

	private:
		int32 width;
		int32 height;
		int32 channels;
		std::vector<uint8> data;
		bool srgb;
	};

}