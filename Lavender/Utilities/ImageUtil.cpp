#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#include "ImageUtil.h"

namespace lavender
{

	void WriteImageToFile(ImageFormat type, char const* filename, uint32 width, uint32 height, void const* data, uint32 stride)
	{
		switch (type)
		{
		case ImageFormat::PNG: stbi_write_png(filename, (int)width, (int)height, 4, data, (int)stride); break;
		case ImageFormat::JPG: stbi_write_jpg(filename, (int)width, (int)height, 4, data, 100); break;
		case ImageFormat::HDR: stbi_write_hdr(filename, (int)width, (int)height, 4, (float const*)data); break;
		case ImageFormat::TGA: stbi_write_tga(filename, (int)width, (int)height, 4, data); break;
		case ImageFormat::BMP: stbi_write_bmp(filename, (int)width, (int)height, 4, data); break;
		default: LAV_UNREACHABLE();
		}
	}
}