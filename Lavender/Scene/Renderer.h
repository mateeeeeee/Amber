#pragma once
#include <memory>

namespace lavender
{
	class Scene;
	class Renderer
	{
	public:
		explicit Renderer(std::unique_ptr<Scene>&& scene);
		~Renderer();

		void Update(float dt);
		void Render();

	private:
		std::unique_ptr<Scene> scene;
	};
}