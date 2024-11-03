#include "Input.h"
#include "Window.h"

namespace amber
{
	static KeyCode ConvertSDLKeycode(SDL_Keycode sdl_keycode)
	{
		switch (sdl_keycode)
		{
		case SDLK_a: return KeyCode::A;
		case SDLK_b: return KeyCode::B;
		case SDLK_c: return KeyCode::C;
		case SDLK_d: return KeyCode::D;
		case SDLK_e: return KeyCode::E;
		case SDLK_f: return KeyCode::F;
		case SDLK_g: return KeyCode::G;
		case SDLK_h: return KeyCode::H;
		case SDLK_i: return KeyCode::I;
		case SDLK_j: return KeyCode::J;
		case SDLK_k: return KeyCode::K;
		case SDLK_l: return KeyCode::L;
		case SDLK_m: return KeyCode::M;
		case SDLK_n: return KeyCode::N;
		case SDLK_o: return KeyCode::O;
		case SDLK_p: return KeyCode::P;
		case SDLK_q: return KeyCode::Q;
		case SDLK_r: return KeyCode::R;
		case SDLK_s: return KeyCode::S;
		case SDLK_t: return KeyCode::T;
		case SDLK_u: return KeyCode::U;
		case SDLK_v: return KeyCode::V;
		case SDLK_w: return KeyCode::W;
		case SDLK_x: return KeyCode::X;
		case SDLK_y: return KeyCode::Y;
		case SDLK_z: return KeyCode::Z;
		}
		return KeyCode::Count;
	}

	void Input::Tick() 
	{
		SDL_PumpEvents();

		prev_keys = std::move(keys);

		prev_mouse_position_x = mouse_position_x;
		prev_mouse_position_y = mouse_position_y;
		Uint32 mouse_state = SDL_GetMouseState(&mouse_position_x, &mouse_position_y);

		using enum KeyCode;

		keys[(Uint64)MouseLeft]		= mouse_state & SDL_BUTTON(1);
		keys[(Uint64)MouseMiddle]	= mouse_state & SDL_BUTTON(2);
		keys[(Uint64)MouseRight]	= mouse_state & SDL_BUTTON(3);

		Uint8 const* sdl_keys = SDL_GetKeyboardState(nullptr);

		keys[(Uint64)F1] = sdl_keys[SDL_SCANCODE_F1];
		keys[(Uint64)F2] = sdl_keys[SDL_SCANCODE_F2];
		keys[(Uint64)F3] = sdl_keys[SDL_SCANCODE_F3];
		keys[(Uint64)F4] = sdl_keys[SDL_SCANCODE_F4];
		keys[(Uint64)F5] = sdl_keys[SDL_SCANCODE_F5];
		keys[(Uint64)F6] = sdl_keys[SDL_SCANCODE_F6];
		keys[(Uint64)F7] = sdl_keys[SDL_SCANCODE_F7];
		keys[(Uint64)F8] = sdl_keys[SDL_SCANCODE_F8];
		keys[(Uint64)F9] = sdl_keys[SDL_SCANCODE_F9];
		keys[(Uint64)F10] = sdl_keys[SDL_SCANCODE_F10];
		keys[(Uint64)F11] = sdl_keys[SDL_SCANCODE_F11];
		keys[(Uint64)F12] = sdl_keys[SDL_SCANCODE_F12];
		keys[(Uint64)Alpha0] = sdl_keys[SDL_SCANCODE_0];
		keys[(Uint64)Alpha1] = sdl_keys[SDL_SCANCODE_1];
		keys[(Uint64)Alpha2] = sdl_keys[SDL_SCANCODE_2];
		keys[(Uint64)Alpha3] = sdl_keys[SDL_SCANCODE_3];
		keys[(Uint64)Alpha4] = sdl_keys[SDL_SCANCODE_4];
		keys[(Uint64)Alpha5] = sdl_keys[SDL_SCANCODE_5];
		keys[(Uint64)Alpha6] = sdl_keys[SDL_SCANCODE_6];
		keys[(Uint64)Alpha7] = sdl_keys[SDL_SCANCODE_7];
		keys[(Uint64)Alpha8] = sdl_keys[SDL_SCANCODE_8];
		keys[(Uint64)Alpha9] = sdl_keys[SDL_SCANCODE_9];
		keys[(Uint64)Numpad0] = sdl_keys[SDL_SCANCODE_KP_0];
		keys[(Uint64)Numpad1] = sdl_keys[SDL_SCANCODE_KP_1];
		keys[(Uint64)Numpad2] = sdl_keys[SDL_SCANCODE_KP_2];
		keys[(Uint64)Numpad3] = sdl_keys[SDL_SCANCODE_KP_3];
		keys[(Uint64)Numpad4] = sdl_keys[SDL_SCANCODE_KP_4];
		keys[(Uint64)Numpad5] = sdl_keys[SDL_SCANCODE_KP_5];
		keys[(Uint64)Numpad6] = sdl_keys[SDL_SCANCODE_KP_6];
		keys[(Uint64)Numpad7] = sdl_keys[SDL_SCANCODE_KP_7];
		keys[(Uint64)Numpad8] = sdl_keys[SDL_SCANCODE_KP_8];
		keys[(Uint64)Numpad9] = sdl_keys[SDL_SCANCODE_KP_9];
		keys[(Uint64)Q] = sdl_keys[SDL_SCANCODE_Q];
		keys[(Uint64)W] = sdl_keys[SDL_SCANCODE_W];
		keys[(Uint64)E] = sdl_keys[SDL_SCANCODE_E];
		keys[(Uint64)R] = sdl_keys[SDL_SCANCODE_R];
		keys[(Uint64)T] = sdl_keys[SDL_SCANCODE_T];
		keys[(Uint64)Y] = sdl_keys[SDL_SCANCODE_Y];
		keys[(Uint64)U] = sdl_keys[SDL_SCANCODE_U];
		keys[(Uint64)I] = sdl_keys[SDL_SCANCODE_I];
		keys[(Uint64)O] = sdl_keys[SDL_SCANCODE_O];
		keys[(Uint64)P] = sdl_keys[SDL_SCANCODE_P];
		keys[(Uint64)A] = sdl_keys[SDL_SCANCODE_A];
		keys[(Uint64)S] = sdl_keys[SDL_SCANCODE_S];
		keys[(Uint64)D] = sdl_keys[SDL_SCANCODE_D];
		keys[(Uint64)F] = sdl_keys[SDL_SCANCODE_F];
		keys[(Uint64)G] = sdl_keys[SDL_SCANCODE_G];
		keys[(Uint64)H] = sdl_keys[SDL_SCANCODE_H];
		keys[(Uint64)J] = sdl_keys[SDL_SCANCODE_J];
		keys[(Uint64)K] = sdl_keys[SDL_SCANCODE_K];
		keys[(Uint64)L] = sdl_keys[SDL_SCANCODE_L];
		keys[(Uint64)Z] = sdl_keys[SDL_SCANCODE_Z];
		keys[(Uint64)X] = sdl_keys[SDL_SCANCODE_X];
		keys[(Uint64)C] = sdl_keys[SDL_SCANCODE_C];
		keys[(Uint64)V] = sdl_keys[SDL_SCANCODE_V];
		keys[(Uint64)B] = sdl_keys[SDL_SCANCODE_B];
		keys[(Uint64)N] = sdl_keys[SDL_SCANCODE_N];
		keys[(Uint64)M] = sdl_keys[SDL_SCANCODE_M];
		keys[(Uint64)Esc] = sdl_keys[SDL_SCANCODE_ESCAPE];
		keys[(Uint64)Tab] = sdl_keys[SDL_SCANCODE_TAB];
		keys[(Uint64)ShiftLeft]		= sdl_keys[SDL_SCANCODE_LSHIFT];
		keys[(Uint64)ShiftRight]	= sdl_keys[SDL_SCANCODE_RSHIFT];
		keys[(Uint64)CtrlLeft]		= sdl_keys[SDL_SCANCODE_LCTRL];
		keys[(Uint64)CtrlRight]		= sdl_keys[SDL_SCANCODE_RCTRL];
		keys[(Uint64)AltLeft]		= sdl_keys[SDL_SCANCODE_LALT];
		keys[(Uint64)AltRight]		= sdl_keys[SDL_SCANCODE_RALT];
		keys[(Uint64)Space]			= sdl_keys[SDL_SCANCODE_SPACE];
		keys[(Uint64)CapsLock]		= sdl_keys[SDL_SCANCODE_CAPSLOCK];
		keys[(Uint64)Backspace]		= sdl_keys[SDL_SCANCODE_BACKSPACE];
		keys[(Uint64)Enter]			= sdl_keys[SDL_SCANCODE_RETURN];
		keys[(Uint64)Delete]		= sdl_keys[SDL_SCANCODE_DELETE];
		keys[(Uint64)ArrowLeft]		= sdl_keys[SDL_SCANCODE_LEFT];
		keys[(Uint64)ArrowRight]	= sdl_keys[SDL_SCANCODE_RIGHT];
		keys[(Uint64)ArrowUp]		= sdl_keys[SDL_SCANCODE_UP];
		keys[(Uint64)ArrowDown]		= sdl_keys[SDL_SCANCODE_DOWN];
		keys[(Uint64)PageUp]		= sdl_keys[SDL_SCANCODE_PAGEUP];
		keys[(Uint64)PageDown]		= sdl_keys[SDL_SCANCODE_PAGEDOWN];
		keys[(Uint64)Home]			= sdl_keys[SDL_SCANCODE_HOME];
		keys[(Uint64)End]			= sdl_keys[SDL_SCANCODE_END];
		keys[(Uint64)Insert]		= sdl_keys[SDL_SCANCODE_INSERT];

	}

	void Input::OnWindowEvent(WindowEventData const& data)
	{
		switch (data.event->type)
		{
		case SDL_WINDOWEVENT:
		{
			if(data.event->window.event == SDL_WINDOWEVENT_RESIZED)
				input_events.window_resized_event.Broadcast(data.event->window.data1, data.event->window.data2);
		}
		break;
		case SDL_KEYDOWN:
		{
			KeyCode keycode = ConvertSDLKeycode(data.event->key.keysym.sym);
			input_events.key_pressed.Broadcast(keycode);
		}
		}
	}

	void Input::SetMouseVisibility(Bool visible)
	{
		SDL_ShowCursor(visible ? SDL_ENABLE : SDL_DISABLE);
	}
}

