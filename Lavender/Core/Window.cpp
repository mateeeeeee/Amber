#include "Window.h"

namespace lavender
{

	Window::Window(uint32 w, uint32 h, char const* title /*= ""*/)
	{
		SDLCheck(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0);
		SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_RESIZABLE);
		window.reset(SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,w, h, window_flags));
		SDLCheck(window.get());
	}

	Window::~Window()
	{
		SDL_Quit();
	}

	WindowDims Window::GetDimensions() const
	{
		int32 w, h;
		SDL_GetWindowSize(window.get(), &w, &h);
		return WindowDims{ w,h };
	}

}

