#pragma once
#include "Utilities/SDLUtil.h"

namespace amber
{
	enum class KeyCode : uint32;
	class Window;
	struct WindowEventData;
	class EditorSink;
	class EditorConsole;
	class Camera;
	class OptixRenderer;

	class Editor
	{
		enum VisibilityFlag
		{
			Visibility_Log,
			Visibility_Console,
			Visibility_Settings,
			Visibility_Stats,
			Visibility_Camera,
			Visibility_Count
		};

	public:
		Editor(Window& window, Camera& camera, OptixRenderer& renderer);
		~Editor();

		void Run();

		void OnResize(int32 w, int32 h);
		void OnWindowEvent(WindowEventData const&);

		void SetEditorSink(EditorSink* sink)
		{
			editor_sink = sink;
		}
		void SetDefaultOptions(uint32 _sample_count, uint32 _max_depth)
		{
			sample_count = _sample_count;
			max_depth = _max_depth;
		}

	private:
		Window& window;
		Camera& camera;
		OptixRenderer& renderer;
		SDLRendererPtr sdl_renderer;
		SDLTexturePtr render_target = nullptr;
		SDLTexturePtr gui_target = nullptr;

		bool gui_enabled = true;
		bool visibility_flags[Visibility_Count] = {false};
		std::unique_ptr<EditorConsole> editor_console;
		EditorSink* editor_sink;

		int32 sample_count = 1;
		int32 max_depth = 1;

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
		void SettingsWindow();
		void CameraWindow();
	};
}