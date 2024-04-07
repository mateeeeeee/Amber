#pragma once
#include <string>

namespace lavender
{
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