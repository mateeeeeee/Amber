#pragma once
#include "Utilities/SDLUtil.h"
#include "Utilities/Delegate.h"

namespace lavender
{
	struct WindowEventData
	{
		SDL_Event* event;
	};
	DECLARE_EVENT(WindowEvent, Window, WindowEventData const&);

	class Window
	{
		friend class Editor;
	public:
		Window(uint32 w, uint32 h, char const* title = "");
		~Window();

		uint32 Width() const;
		uint32 Height() const;

		bool Loop();
		WindowEvent& GetWindowEvent() { return window_event; }

	private:
		SDLWindowPtr sdl_window = nullptr;
		uint32 width, height;
		WindowEvent window_event;
	};
}