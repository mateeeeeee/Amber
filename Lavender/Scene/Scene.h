#pragma once
#include <memory>
#include <vector>
#include "Mesh.h"
#include "Material.h"
#include "Utilities/ImageUtil.h"

namespace lavender
{
	
	struct Scene
	{
		std::vector<Mesh> meshes;
		std::vector<Instance> instances;
		std::vector<Material> materials;
		std::vector<Image> textures;
		BoundingBox bounding_box;

		void Merge(std::unique_ptr<Scene>& s)
		{
			s.reset();
		}
	};

	std::unique_ptr<Scene> LoadScene(char const* scene_file);
}