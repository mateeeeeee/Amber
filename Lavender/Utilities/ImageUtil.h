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
}