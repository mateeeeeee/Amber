#include "Platform/Input.h"

namespace amber
{
	void Input::Tick()
	{
		prev_keys = keys;
		prev_mouse_position_x = mouse_position_x;
		prev_mouse_position_y = mouse_position_y;
		mouse_wheel_delta = 0.0f;
		new_frame = true;
	}

	void Input::OnWindowEvent(WindowEventData const&)
	{
		// No-op on Mac - event handling is done directly through NSEvent
	}

	void Input::SetMouseVisibility(Bool visible)
	{

	}
}
