#pragma once
#include "Utilities/SDLUtil.h"

namespace lavender
{
	class Window;
	struct WindowEventData;
	enum class KeyCode : uint32;

	class Editor
	{
	public:
		Editor(Window& window);
		~Editor();

		void Run();

		void OnResize(int32 w, int32 h);
		void OnWindowEvent(WindowEventData const&);

	private:
		Window& window;
		SDLRendererPtr renderer;
		SDLTexturePtr render_target = nullptr;
		SDLTexturePtr gui_target = nullptr;

		bool gui_enabled = true;

	private:
		void SetStyle();

		void OnKeyPressed(KeyCode keycode);

		void Begin();
		void Render();
		void End();

		void BeginGUI();
		void GUI();
		void EndGUI();
	};
}