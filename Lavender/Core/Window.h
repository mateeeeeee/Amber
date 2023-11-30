#pragma once
#include <string>
#include "Utilities/SDLUtil.h"

namespace lavender
{
	struct WindowDims
	{
		int32 x;
		int32 y;
	};

	class Window
	{
	public:
		Window(uint32 w, uint32 h, char const* title = "");
		~Window();

		WindowDims GetDimensions() const;



	private:
		SDLWindowPtr window = nullptr;
	};
}