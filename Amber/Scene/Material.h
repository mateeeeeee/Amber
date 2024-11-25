#pragma once
#include <string>

namespace amber
{
	struct Material 
	{
		Vector3 base_color = Vector3(0.9f, 0.9f, 0.9f);
		Vector3 emissive_color = Vector3(0.0f, 0.0f, 0.0f);
		Float metallic = 0.0f;
		Float specular = 0.0f;
		Float roughness = 1.0f;
		Float specular_tint = 0.0f;
		Float anisotropy = 0.0f;
		Float alpha_cutoff = 0.5f;

		Float sheen = 0.0f;
		Float sheen_tint = 0.0f;
		Float clearcoat = 0.0f;
		Float clearcoat_gloss = 0.0f;

		Float ior = 1.5f;
		Float specular_transmission = 0.0f;

		Int32 diffuse_tex_id = -1;
		Int32 metallic_roughness_tex_id = -1;
		Int32 normal_tex_id = -1;
		Int32 emissive_tex_id = -1;
	};
}