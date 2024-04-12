#pragma once

namespace lavender
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

	struct Image
	{
		explicit Image(char const* file, bool srgb = false);

		int32 width;
		int32 height;
		int32 channels;
		std::vector<uint8> data;
		bool srgb;
	};

}