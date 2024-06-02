#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

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
		default: LAV_ASSERT(false);
		}
	}


	Image::Image(char const* file, bool srgb) : srgb(srgb)
	{
		stbi_set_flip_vertically_on_load(1);
		uint8* image_data = stbi_load(file, &width, &height, &channels, 4);
		channels = 4;
		LAV_ASSERT_MSG(image_data, "Could not load image");
		data = std::vector<uint8>(image_data, image_data + width * height * channels);
		stbi_image_free(image_data);
		stbi_set_flip_vertically_on_load(0);
	}
}