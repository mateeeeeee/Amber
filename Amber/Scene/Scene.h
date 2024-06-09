#pragma once
#include <memory>
#include <vector>
#include "Mesh.h"
#include "Material.h"
#include "Utilities/ImageUtil.h"

namespace amber
{
	struct Scene
	{
		std::vector<Mesh> meshes;
		std::vector<Instance> instances;
		std::vector<Material> materials;
		std::vector<Image> textures;
		std::unique_ptr<Image> environment;

		void Merge(std::unique_ptr<Scene>& s)
		{
			//#todo 
		}
	};

	std::unique_ptr<Scene> LoadScene(char const* scene_file, char const* environment_texture, float scale = 1.0f);
}