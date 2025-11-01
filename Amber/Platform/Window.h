#pragma once
#include "Utilities/SDLUtil.h"
#include "Utilities/Delegate.h"

namespace amber
{
	class Editor;

	struct WindowEventData
	{
		SDL_Event* event;
	};
	DECLARE_EVENT(WindowEvent, Window, WindowEventData const&)

	class Window
	{
		friend class Editor;
	public:
		Window(Uint32 w, Uint32 h, Char const* title = "");
		~Window();

		Uint32 Width() const;
		Uint32 Height() const;

		void Maximize();
		Bool Loop();
		WindowEvent& GetWindowEvent() { return window_event; }

	private:
		SDLWindowPtr sdl_window = nullptr;
		Uint32 width, height;
		WindowEvent window_event;
	};
}