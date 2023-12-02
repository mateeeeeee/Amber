#include "SDLUtil.h"
#include "Core/Logger.h"

namespace lavender
{
	void SDLCheck(int32 r)
	{
		if (r != 0)
		{
			LAVENDER_ERROR("{}", SDL_GetError());
			std::exit(1);
		}
	}
	void SDLCheck(void* sdl_type)
	{
		if (sdl_type == nullptr)
		{
			LAVENDER_ERROR("{}", SDL_GetError());
			std::exit(1);
		}
	}
}