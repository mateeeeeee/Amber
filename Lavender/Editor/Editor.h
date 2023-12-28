#pragma once
#include "Utilities/SDLUtil.h"

namespace lavender
{
	class Window;
	struct WindowEventData;
	enum class KeyCode : uint32;
	class EditorSink;
	class EditorConsole;

	class Editor
	{
		enum VisibilityFlag
		{
			Visibility_Log,
			Visibility_Console,
			Visibility_Settings,
			Visibility_Count
		};

	public:
		Editor(Window& window, EditorSink& editor_sink);
		~Editor();

		void Run();

		void OnResize(int32 w, int32 h);
		void OnWindowEvent(WindowEventData const&);

	private:
		Window& window;
		SDLRendererPtr renderer;
		SDLTexturePtr render_target = nullptr;
		SDLTexturePtr gui_target = nullptr;

		bool gui_enabled = true;
		bool visibility_flags[Visibility_Count] = {false};
		std::unique_ptr<EditorConsole> editor_console;
		EditorSink& editor_sink;

	private:
		void SetStyle();

		void OnKeyPressed(KeyCode keycode);

		void Begin();
		void Render();
		void End();

		void BeginGUI();
		void GUI();
		void EndGUI();

		void LogWindow();
		void ConsoleWindow();
	};
}