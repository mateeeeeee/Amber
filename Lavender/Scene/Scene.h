#pragma once
#include <memory>
#include <optional>

namespace lavender
{
	class Camera;
	struct Scene
	{
		std::unique_ptr<Camera> camera;
	};
	std::optional<Scene> LoadScene(char const* scene_file);
}