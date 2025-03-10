#pragma once
#include <array>
#include "Utilities/Delegate.h"
#include "Utilities/Singleton.h"

namespace amber
{
	enum class KeyCode : Uint32
	{
		F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
		Alpha0, Alpha1, Alpha2, Alpha3, Alpha4, Alpha5, Alpha6, Alpha7, Alpha8, Alpha9,
		Numpad0, Numpad1, Numpad2, Numpad3, Numpad4, Numpad5, Numpad6, Numpad7, Numpad8, Numpad9,
		Q, W, E, R, T, Y, U, I, O, P,
		A, S, D, F, G, H, J, K, L,
		Z, X, C, V, B, N, M,
		Esc,
		Tab,
		ShiftLeft, ShiftRight,
		CtrlLeft, CtrlRight,
		AltLeft, AltRight,
		Space,
		CapsLock,
		Backspace,
		Enter,
		Delete,
		ArrowLeft, ArrowRight, ArrowUp, ArrowDown,
		PageUp, PageDown,
		Home,
		End,
		Insert,
		MouseLeft,
		MouseMiddle,
		MouseRight,
		Count
	};

	struct WindowEventData;
	DECLARE_EVENT(WindowResizedEvent, Input, Int32, Int32)
	DECLARE_EVENT(KeyPressedEvent, Input, KeyCode)

	struct InputEvents
	{
		WindowResizedEvent window_resized_event;
		KeyPressedEvent		   key_pressed;
	};

	class Input : public Singleton<Input>
	{
		friend class Singleton<Input>;

	public:
		InputEvents& GetInputEvents() { return input_events; }

		void Tick();
		void OnWindowEvent(WindowEventData const&);

		Bool GetKey(KeyCode key)    const { return keys[(Uint64)key]; }
		Bool IsKeyDown(KeyCode key) const { return GetKey(key) && !prev_keys[(Uint64)key]; }
		Bool IsKeyUp(KeyCode key)   const { return !GetKey(key) && prev_keys[(Uint64)key]; }

		void SetMouseVisibility(Bool visible);

		Int   GetMousePositionX()  const { return mouse_position_x; }
		Int   GetMousePositionY()  const { return mouse_position_y; }

		Int   GetMouseDeltaX()     const { return mouse_position_x - prev_mouse_position_x; }
		Int   GetMouseDeltaY()     const { return mouse_position_y - prev_mouse_position_y; }
		Float GetMouseWheelDelta() const { return mmouse_wheel_delta; }

	private:
		InputEvents input_events;
		std::array<Bool, (Uint64)KeyCode::Count> keys = {};
		std::array<Bool, (Uint64)KeyCode::Count> prev_keys = {};
		
		Int   mouse_position_x = 0;
		Int   mouse_position_y = 0;

		Int   prev_mouse_position_x = 0;
		Int   prev_mouse_position_y = 0;
		Float mmouse_wheel_delta = 0.0f;

		Bool new_frame = false;
		Bool resizing = false;

	private:
		Input() = default;
	};
	#define g_Input Input::Get()

}