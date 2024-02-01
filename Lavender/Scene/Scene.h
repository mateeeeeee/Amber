#pragma once
#include <memory>
#include <vector>
#include "Mesh.h"
#include "Material.h"

namespace lavender
{
	
	struct Scene
	{
		std::vector<Mesh> meshes;
		std::vector<Primitive> primitives;
		std::vector<Instance> instances;
		std::vector<Material> materials;
		std::vector<Image> textures;
		//std::vector<QuadLight> lights;
		BoundingBox bounding_box;
	};

	std::unique_ptr<Scene> LoadScene(char const* scene_file);
}