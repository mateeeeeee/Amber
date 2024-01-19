#pragma once
#include <memory>
#include <vector>
#include "Light.h"
#include "Primitive.h"

namespace lavender
{
	struct Scene
	{
		std::vector<Light> lights;
		std::vector<Primitive> primitives;
		BoundingBox bounding_box;
	};

	std::unique_ptr<Scene> LoadScene(char const* scene_file);
}