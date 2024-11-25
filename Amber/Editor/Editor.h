#pragma once
#include "Utilities/SDLUtil.h"

namespace amber
{
	enum class KeyCode : Uint32;
	class Window;
	struct WindowEventData;
	class EditorSink;
	class EditorConsole;
	class Camera;
	class OptixPathTracer;

	class Editor
	{
		enum VisibilityFlag
		{
			Visibility_Log,
			Visibility_Console,
			Visibility_Options,
			Visibility_Debug,
			Visibility_Lights,
			Visibility_Stats,
			Visibility_Camera,
			Visibility_Count
		};

	public:
		Editor(Window& window, Camera& camera, OptixPathTracer& path_tracer);
		~Editor();

		void Run();

		void OnResize(Int32 w, Int32 h);
		void OnWindowEvent(WindowEventData const&);

		void SetEditorSink(EditorSink* sink)
		{
			editor_sink = sink;
		}

	private:
		Window& window;
		Camera& camera;
		OptixPathTracer& path_tracer;
		SDLRendererPtr sdl_renderer;
		SDLTexturePtr render_target = nullptr;
		SDLTexturePtr gui_target = nullptr;

		Bool gui_enabled = true;
		Bool scene_focused = false;
		Bool visibility_flags[Visibility_Count] = {false};
		std::unique_ptr<EditorConsole> editor_console;
		EditorSink* editor_sink;

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
		void StatsWindow();
		void OptionsWindow();
		void DebugWindow();
		void CameraWindow();
		void LightsWindow();
	};
}