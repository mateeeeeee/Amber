#pragma once
#include <memory>
#include <vector>

namespace lavender
{
	

	struct Scene
	{
		BoundingBox bounding_box;
	};

	std::unique_ptr<Scene> LoadScene(char const* scene_file);
}