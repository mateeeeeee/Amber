#pragma once
#include <memory>

namespace lavender
{
	struct Scene
	{
		uint32 primitive_count;
	};
	std::unique_ptr<Scene> LoadScene(char const* scene_file);
}