#pragma once
#include <SDL.h>
#include <memory>

namespace amber
{
	struct SDLWindowDestroyer
	{
		void operator()(SDL_Window* window) const
		{
			SDL_DestroyWindow(window);
		}
	};
	using SDLWindowPtr = std::unique_ptr<SDL_Window, SDLWindowDestroyer>;

	struct SDLRendererDestroyer
	{
		void operator()(SDL_Renderer* renderer) const
		{
			SDL_DestroyRenderer(renderer);
		}
	};
	using SDLRendererPtr = std::unique_ptr<SDL_Renderer, SDLRendererDestroyer>;

	struct SDLTextureDestroyer
	{
		void operator()(SDL_Texture* texture) const
		{
			SDL_DestroyTexture(texture);
		}
	};
	using SDLTexturePtr = std::unique_ptr<SDL_Texture, SDLTextureDestroyer>;

	struct SDLSurfaceFreer
	{
		void operator()(SDL_Surface* surface) const
		{
			SDL_FreeSurface(surface);
		}
	};
	using SDLSurfacePtr = std::unique_ptr<SDL_Surface, SDLSurfaceFreer>;

	void SDLCheck(Int32 r);
	void SDLCheck(void* sdl_type);
}