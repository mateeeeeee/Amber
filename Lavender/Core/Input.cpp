#include "Input.h"
#include "Window.h"

namespace lavender
{

	void Input::Tick() 
	{
		SDL_PumpEvents();

		prev_keys = std::move(keys);

		prev_mouse_position_x = mouse_position_x;
		prev_mouse_position_y = mouse_position_y;
		uint32 mouse_state = SDL_GetMouseState(&mouse_position_x, &mouse_position_y);

		using enum KeyCode;

		keys[(uint64)MouseLeft]		= mouse_state & SDL_BUTTON(1);
		keys[(uint64)MouseMiddle]	= mouse_state & SDL_BUTTON(2);
		keys[(uint64)MouseRight]	= mouse_state & SDL_BUTTON(3);

		uint8 const* sdl_keys = SDL_GetKeyboardState(nullptr);

		keys[(uint64)F1] = sdl_keys[SDL_SCANCODE_F1];
		keys[(uint64)F2] = sdl_keys[SDL_SCANCODE_F2];
		keys[(uint64)F3] = sdl_keys[SDL_SCANCODE_F3];
		keys[(uint64)F4] = sdl_keys[SDL_SCANCODE_F4];
		keys[(uint64)F5] = sdl_keys[SDL_SCANCODE_F5];
		keys[(uint64)F6] = sdl_keys[SDL_SCANCODE_F6];
		keys[(uint64)F7] = sdl_keys[SDL_SCANCODE_F7];
		keys[(uint64)F8] = sdl_keys[SDL_SCANCODE_F8];
		keys[(uint64)F9] = sdl_keys[SDL_SCANCODE_F9];
		keys[(uint64)F10] = sdl_keys[SDL_SCANCODE_F10];
		keys[(uint64)F11] = sdl_keys[SDL_SCANCODE_F11];
		keys[(uint64)F12] = sdl_keys[SDL_SCANCODE_F12];
		keys[(uint64)Alpha0] = sdl_keys[SDL_SCANCODE_0];
		keys[(uint64)Alpha1] = sdl_keys[SDL_SCANCODE_1];
		keys[(uint64)Alpha2] = sdl_keys[SDL_SCANCODE_2];
		keys[(uint64)Alpha3] = sdl_keys[SDL_SCANCODE_3];
		keys[(uint64)Alpha4] = sdl_keys[SDL_SCANCODE_4];
		keys[(uint64)Alpha5] = sdl_keys[SDL_SCANCODE_5];
		keys[(uint64)Alpha6] = sdl_keys[SDL_SCANCODE_6];
		keys[(uint64)Alpha7] = sdl_keys[SDL_SCANCODE_7];
		keys[(uint64)Alpha8] = sdl_keys[SDL_SCANCODE_8];
		keys[(uint64)Alpha9] = sdl_keys[SDL_SCANCODE_9];
		keys[(uint64)Numpad0] = sdl_keys[SDL_SCANCODE_KP_0];
		keys[(uint64)Numpad1] = sdl_keys[SDL_SCANCODE_KP_1];
		keys[(uint64)Numpad2] = sdl_keys[SDL_SCANCODE_KP_2];
		keys[(uint64)Numpad3] = sdl_keys[SDL_SCANCODE_KP_3];
		keys[(uint64)Numpad4] = sdl_keys[SDL_SCANCODE_KP_4];
		keys[(uint64)Numpad5] = sdl_keys[SDL_SCANCODE_KP_5];
		keys[(uint64)Numpad6] = sdl_keys[SDL_SCANCODE_KP_6];
		keys[(uint64)Numpad7] = sdl_keys[SDL_SCANCODE_KP_7];
		keys[(uint64)Numpad8] = sdl_keys[SDL_SCANCODE_KP_8];
		keys[(uint64)Numpad9] = sdl_keys[SDL_SCANCODE_KP_9];
		keys[(uint64)Q] = sdl_keys[SDLK_q];
		keys[(uint64)W] = sdl_keys[SDLK_w];
		keys[(uint64)E] = sdl_keys[SDLK_e];
		keys[(uint64)R] = sdl_keys[SDLK_r];
		keys[(uint64)T] = sdl_keys[SDLK_t];
		keys[(uint64)Y] = sdl_keys[SDLK_y];
		keys[(uint64)U] = sdl_keys[SDLK_u];
		keys[(uint64)I] = sdl_keys[SDLK_i];
		keys[(uint64)O] = sdl_keys[SDLK_o];
		keys[(uint64)P] = sdl_keys[SDLK_p];
		keys[(uint64)A] = sdl_keys[SDLK_a];
		keys[(uint64)S] = sdl_keys[SDLK_s];
		keys[(uint64)D] = sdl_keys[SDLK_d];
		keys[(uint64)F] = sdl_keys[SDLK_f];
		keys[(uint64)G] = sdl_keys[SDLK_g];
		keys[(uint64)H] = sdl_keys[SDLK_h];
		keys[(uint64)J] = sdl_keys[SDLK_j];
		keys[(uint64)K] = sdl_keys[SDLK_k];
		keys[(uint64)L] = sdl_keys[SDLK_l];
		keys[(uint64)Z] = sdl_keys[SDLK_z];
		keys[(uint64)X] = sdl_keys[SDLK_x];
		keys[(uint64)C] = sdl_keys[SDLK_c];
		keys[(uint64)V] = sdl_keys[SDLK_v];
		keys[(uint64)B] = sdl_keys[SDLK_b];
		keys[(uint64)N] = sdl_keys[SDLK_n];
		keys[(uint64)M] = sdl_keys[SDLK_m];
		keys[(uint64)Esc] = sdl_keys[SDL_SCANCODE_ESCAPE];
		keys[(uint64)Tab] = sdl_keys[SDL_SCANCODE_TAB];
		keys[(uint64)ShiftLeft]		= sdl_keys[SDL_SCANCODE_LSHIFT];
		keys[(uint64)ShiftRight]	= sdl_keys[SDL_SCANCODE_RSHIFT];
		keys[(uint64)CtrlLeft]		= sdl_keys[SDL_SCANCODE_LCTRL];
		keys[(uint64)CtrlRight]		= sdl_keys[SDL_SCANCODE_RCTRL];
		keys[(uint64)AltLeft]		= sdl_keys[SDL_SCANCODE_LALT];
		keys[(uint64)AltRight]		= sdl_keys[SDL_SCANCODE_RALT];
		keys[(uint64)Space]			= sdl_keys[SDL_SCANCODE_SPACE];
		keys[(uint64)CapsLock]		= sdl_keys[SDL_SCANCODE_CAPSLOCK];
		keys[(uint64)Backspace]		= sdl_keys[SDL_SCANCODE_BACKSPACE];
		keys[(uint64)Enter]			= sdl_keys[SDL_SCANCODE_RETURN];
		keys[(uint64)Delete]		= sdl_keys[SDL_SCANCODE_DELETE];
		keys[(uint64)ArrowLeft]		= sdl_keys[SDL_SCANCODE_LEFT];
		keys[(uint64)ArrowRight]	= sdl_keys[SDL_SCANCODE_RIGHT];
		keys[(uint64)ArrowUp]		= sdl_keys[SDL_SCANCODE_UP];
		keys[(uint64)ArrowDown]		= sdl_keys[SDL_SCANCODE_DOWN];
		keys[(uint64)PageUp]		= sdl_keys[SDL_SCANCODE_PAGEUP];
		keys[(uint64)PageDown]		= sdl_keys[SDL_SCANCODE_PAGEDOWN];
		keys[(uint64)Home]			= sdl_keys[SDL_SCANCODE_HOME];
		keys[(uint64)End]			= sdl_keys[SDL_SCANCODE_END];
		keys[(uint64)Insert]		= sdl_keys[SDL_SCANCODE_INSERT];
	}

	void Input::OnWindowEvent(WindowEventData const& data)
	{
		switch (data.event->type)
		{
		case SDL_WINDOWEVENT_RESIZED:
		{
			input_events.window_resized_event.Broadcast(data.event->window.data1, data.event->window.data2);
		}
		break;
		case SDL_MOUSEWHEEL: 
		break;
		}
	}

	void Input::SetMouseVisibility(bool visible)
	{
		SDL_ShowCursor(visible ? SDL_ENABLE : SDL_DISABLE);
	}
}

