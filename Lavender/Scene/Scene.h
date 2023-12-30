#pragma once
#include <memory>

namespace lavender
{
	struct Scene
	{
	};
	std::unique_ptr<Scene> LoadScene(char const* scene_file);
}