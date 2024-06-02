#include "SDLUtil.h"
#include "Core/Logger.h"

namespace amber
{
	void SDLCheck(int32 r)
	{
		if (r != 0)
		{
			LAV_ERROR("{}", SDL_GetError());
			std::exit(1);
		}
	}
	void SDLCheck(void* sdl_type)
	{
		if (sdl_type == nullptr)
		{
			LAV_ERROR("{}", SDL_GetError());
			std::exit(1);
		}
	}
}