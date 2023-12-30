#pragma once

namespace lavender
{
	class Renderer
	{
	public:
		explicit Renderer(char const* config_file);

		void Update(float dt);
		void Render();
	};
}