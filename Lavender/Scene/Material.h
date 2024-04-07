#pragma once
#include <string>

namespace lavender
{
	//move to image.h/cpp
	struct Image 
	{
		std::string name;
		uint32 width;
		uint32 height;
		uint32 channels;
		std::vector<uint8> img;

		explicit Image(std::string_view file);
		Image(uint8 const* buf,
			int width, int height, int channels,
			std::string_view name);
	};

	struct Material 
	{
		Vector3 base_color = Vector3(0.9f, 0.9f, 0.9f);
		float metallic = 0.0f;

		float specular = 0.0f;
		float roughness = 1.0f;
		float specular_tint = 0.0f;
		float anisotropy = 0.0f;

		float sheen = 0.0f;
		float sheen_tint = 0.0f;
		float clearcoat = 0.0f;
		float clearcoat_gloss = 0.0f;

		float ior = 1.5f;
		float specular_transmission = 0.0f;
	};
}