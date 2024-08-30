#pragma once
#include <string>

namespace amber
{
	struct Material 
	{
		Vector3 base_color = Vector3(0.9f, 0.9f, 0.9f);
		Vector3 emissive_color = Vector3(0.0f, 0.0f, 0.0f);
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

		int32 diffuse_tex_id = -1;
		int32 emissive_tex_id = -1;
	};
}