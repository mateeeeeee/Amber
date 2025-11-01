#pragma once
#include <string>
#include "SDLUtil.h"
#include "Editor/IImGuiManager.h"

namespace amber
{
	class Window;

	class SDLImGuiManager : public IImGuiManager
	{
	public:
		SDLImGuiManager(Window& window, SDLRendererPtr const& renderer);
		~SDLImGuiManager() override;

		void Initialize() override;
		void Shutdown() override;
		void BeginFrame() override;
		void EndFrame() override;
		void Render() override;

	private:
		Window& window;
		SDLRendererPtr const& renderer;
		std::string ini_file;
	};
}
