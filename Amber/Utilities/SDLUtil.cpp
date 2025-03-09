#include "SDLUtil.h"
#include "Core/Log.h"

namespace amber
{
	void SDLCheck(Int32 r)
	{
		if (r != 0)
		{
			AMBER_ERROR("%s", SDL_GetError());
			std::exit(1);
		}
	}
	void SDLCheck(void* sdl_type)
	{
		if (sdl_type == nullptr)
		{
			AMBER_ERROR("%s", SDL_GetError());
			std::exit(1);
		}
	}
}