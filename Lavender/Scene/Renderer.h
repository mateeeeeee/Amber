#pragma once
#include <memory>
#include "Utilities/Buffer2D.h"

namespace lavender
{
	class Scene;
	class Camera;

	class Renderer
	{
		using Framebuffer = Buffer2D<Vector4>;
	public:
		explicit Renderer(uint32 width, uint32 height, std::unique_ptr<Scene>&& scene);
		~Renderer();

		void Update(float dt);
		void Render(Camera const& camera);

		void OnResize(uint32 w, uint32 h);

		void WriteFramebuffer(char const* outfile);

	private:
		Framebuffer framebuffer;
	};
}