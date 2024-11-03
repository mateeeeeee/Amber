#include "Window.h"
#include "Core/Paths.h"

namespace amber
{

	Window::Window(Uint32 w, Uint32 h, Char const* title) : width(w), height(h)
	{
		SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_RESIZABLE);
		sdl_window.reset(SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, w, h, window_flags));
		SDLCheck(sdl_window.get());
		std::string icon_path = paths::IconsDir + "amberlogo.bmp";
		SDL_Surface* icon = SDL_LoadBMP(icon_path.c_str());
		SDL_SetWindowIcon(sdl_window.get(), icon);
	}
	Window::~Window() = default;

	Uint32 Window::Width() const
	{
		return width;
	}
	Uint32 Window::Height() const
	{
		return height;
	}

	void Window::Maximize()
	{
		SDL_MaximizeWindow(sdl_window.get());
	}

	Bool Window::Loop()
	{
		SDL_Event event;
		while (SDL_PollEvent(&event))
		{
			if (event.type == SDL_QUIT) return false;
			if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE
				&& event.window.windowID == SDL_GetWindowID(sdl_window.get())) return false;
			if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_RESIZED
				&& event.window.windowID == SDL_GetWindowID(sdl_window.get()))
			{
				width = event.window.data1;
				height = event.window.data2;
			}
			//if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_MAXIMIZED
			//	&& event.window.windowID == SDL_GetWindowID(sdl_window.get()))
			//{
			//	width = event.window.data1;
			//	height = event.window.data2;
			//}
			window_event.Broadcast(WindowEventData{ &event });
		}
		return true;
	}

}

