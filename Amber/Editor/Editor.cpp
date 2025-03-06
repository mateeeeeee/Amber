#include "Editor.h"
#include "EditorConsole.h"
#include "EditorSink.h"
#include "Core/Window.h"
#include "Core/Input.h"
#include "Core/Log.h"
#include "Core/Paths.h"
#include "Scene/Camera.h"
#include "Optix/OptixPathTracer.h"
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_sdl.h"
#include "ImGui/imgui_impl_sdlrenderer.h"
#include "FontAwesome/IconsFontAwesome6.h"

namespace amber
{

	Editor::Editor(Window& window, Camera& camera, OptixPathTracer& path_tracer)
		: window(window), camera(camera), path_tracer(path_tracer)
	{
		SDLCheck(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0);
		window.GetWindowEvent().AddMember(&Editor::OnWindowEvent, *this);
		g_Input.GetInputEvents().key_pressed.AddMember(&Editor::OnKeyPressed, *this);
		g_Input.GetInputEvents().window_resized_event.AddMember(&Editor::OnResize, *this);

		editor_console = std::make_unique<EditorConsole>();

		Uint32 renderer_flags = SDL_RENDERER_ACCELERATED;
		renderer_flags |= SDL_RENDERER_PRESENTVSYNC;
		sdl_renderer.reset(SDL_CreateRenderer(window.sdl_window.get(), -1, renderer_flags));
		SDLCheck(sdl_renderer.get());

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		//ini_file = paths::IniDir + "imgui.ini";
		//io.IniFilename = ini_file.c_str();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
		io.ConfigWindowsResizeFromEdges = true;
		ImGui::StyleColorsDark();
		ImGui_ImplSDL2_InitForSDLRenderer(window.sdl_window.get(), sdl_renderer.get());
		ImGui_ImplSDLRenderer_Init(sdl_renderer.get());

		ImFontConfig font_config{};
		std::string font_name = paths::FontsDir + "roboto/Roboto-Light.ttf";
		io.Fonts->AddFontFromFileTTF(font_name.c_str(), 16.0f, &font_config);
		font_config.MergeMode = true;
		ImWchar const icon_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
		std::string icon_name = paths::FontsDir + "FontAwesome/" FONT_ICON_FILE_NAME_FAS;
		io.Fonts->AddFontFromFileTTF(icon_name.c_str(), 15.0f, &font_config, icon_ranges);
		io.Fonts->Build();

		render_target.reset(SDL_CreateTexture(sdl_renderer.get(),
			SDL_PIXELFORMAT_RGBA32,
			SDL_TEXTUREACCESS_STREAMING,
			window.Width(), window.Height()));
		SDLCheck(render_target.get());

		gui_target.reset(SDL_CreateTexture(
						 sdl_renderer.get(), 
			SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_TARGET,
						 window.Width(), window.Height()));
		SDLCheck(gui_target.get());
		SetStyle();
		visibility_flags[Visibility_Scene] = true;
	}

	Editor::~Editor()
	{
		ImGui_ImplSDLRenderer_Shutdown();
		ImGui_ImplSDL2_Shutdown();
		ImGui::DestroyContext();
		SDL_Quit();
	}

	void Editor::Run()
	{
		g_Input.Tick();
		if (gui_enabled) camera.Enable(scene_focused);
		else camera.Enable(true);

		Float dt = ImGui::GetIO().DeltaTime;
		camera.Update(dt);
		path_tracer.Update(dt);
		Begin();
		{
			Render();
			if (gui_enabled)
			{
				BeginGUI();
				GUI();
				EndGUI();
			}
			else
			{
				SDLCheck(SDL_RenderCopy(sdl_renderer.get(), render_target.get(), nullptr, nullptr));
			}
		}
		End();
	}

	void Editor::OnResize(Int32 w, Int32 h)
	{
		render_target.reset(SDL_CreateTexture(sdl_renderer.get(),
			SDL_PIXELFORMAT_RGBA32,
			SDL_TEXTUREACCESS_STREAMING,
			window.Width(), window.Height()));
		SDLCheck(render_target.get());

		gui_target.reset(SDL_CreateTexture(sdl_renderer.get(),
			SDL_PIXELFORMAT_RGBA32,
			SDL_TEXTUREACCESS_TARGET,
			w, h));
		SDLCheck(gui_target.get());

		path_tracer.OnResize(w, h);
		camera.SetAspectRatio((Float)w / h);
	}

	void Editor::OnWindowEvent(WindowEventData const& data)
	{
		g_Input.OnWindowEvent(data);
		if (gui_enabled) ImGui_ImplSDL2_ProcessEvent(data.event);
	}

	void Editor::SetStyle()
	{
		ImGuiStyle& style = ImGui::GetStyle();

		style.Alpha = 1.0f;
		style.FrameRounding = 3.0f;
		style.Colors[ImGuiCol_Text] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
		style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
		style.Colors[ImGuiCol_WindowBg] = ImVec4(0.94f, 0.94f, 0.94f, 0.94f);
		style.Colors[ImGuiCol_PopupBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.94f);
		style.Colors[ImGuiCol_Border] = ImVec4(0.00f, 0.00f, 0.00f, 0.39f);
		style.Colors[ImGuiCol_BorderShadow] = ImVec4(1.00f, 1.00f, 1.00f, 0.10f);
		style.Colors[ImGuiCol_FrameBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.94f);
		style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
		style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
		style.Colors[ImGuiCol_TitleBg] = ImVec4(0.96f, 0.96f, 0.96f, 1.00f);
		style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(1.00f, 1.00f, 1.00f, 0.51f);
		style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.82f, 0.82f, 0.82f, 1.00f);
		style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
		style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.98f, 0.98f, 0.98f, 0.53f);
		style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.69f, 0.69f, 0.69f, 1.00f);
		style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.59f, 0.59f, 0.59f, 1.00f);
		style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.49f, 0.49f, 0.49f, 1.00f);
		style.Colors[ImGuiCol_CheckMark] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
		style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.24f, 0.52f, 0.88f, 1.00f);
		style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
		style.Colors[ImGuiCol_Button] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
		style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
		style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.06f, 0.53f, 0.98f, 1.00f);
		style.Colors[ImGuiCol_Header] = ImVec4(0.26f, 0.59f, 0.98f, 0.31f);
		style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
		style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
		style.Colors[ImGuiCol_ResizeGrip] = ImVec4(1.00f, 1.00f, 1.00f, 0.50f);
		style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
		style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
		style.Colors[ImGuiCol_PlotLines] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
		style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
		style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
		style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
		style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
		
		for (int i = 0; i <= ImGuiCol_COUNT; i++)
		{
			ImVec4& col = style.Colors[i];
			Float H, S, V;
			ImGui::ColorConvertRGBtoHSV(col.x, col.y, col.z, H, S, V);

			if (S < 0.1f)
			{
				V = 1.0f - V;
			}
			ImGui::ColorConvertHSVtoRGB(H, S, V, col.x, col.y, col.z);
			if (col.w < 1.00f)
			{
				col.w *= 0.9f;
			}
		}
	}

	void Editor::OnKeyPressed(KeyCode keycode)
	{
		if (scene_focused && keycode == KeyCode::I)
		{
			gui_enabled = !gui_enabled;
			g_Input.SetMouseVisibility(gui_enabled);
		}
		if (keycode == KeyCode::F12)
		{
			path_tracer.WriteFramebuffer("screenshot");
		}
	}

	void Editor::Begin()
	{
		SDL_SetRenderDrawColor(sdl_renderer.get(), 0, 0, 0, SDL_ALPHA_OPAQUE);
		SDL_RenderClear(sdl_renderer.get());
	}

	void Editor::Render()
	{
		path_tracer.Render(camera);
		auto const& fb = path_tracer.GetFramebuffer();

		int width, height, pitch = -1; void* data = nullptr;
		SDL_QueryTexture(render_target.get(), nullptr, nullptr, &width, &height);
		SDLCheck(SDL_LockTexture(render_target.get(), nullptr, (void**)&data, &pitch));
		memcpy(data, fb, pitch * height);
		SDL_UnlockTexture(render_target.get());
	}

	void Editor::End()
	{
		SDL_RenderPresent(sdl_renderer.get());
	}

	void Editor::BeginGUI()
	{
		ImGui_ImplSDLRenderer_NewFrame();
		ImGui_ImplSDL2_NewFrame();
		ImGui::NewFrame();

		SDLCheck(SDL_SetRenderTarget(sdl_renderer.get(), gui_target.get()));
		SDL_RenderClear(sdl_renderer.get());
		ImGuiID dockspace_id = ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);
	}

	void Editor::GUI()
	{
		if (ImGui::BeginMainMenuBar())
		{
			if (ImGui::BeginMenu(ICON_FA_WINDOW_MAXIMIZE" Windows"))
			{
				if (ImGui::MenuItem(ICON_FA_GLOBE" Scene", 0, visibility_flags[Visibility_Scene]))
					visibility_flags[Visibility_Scene] = !visibility_flags[Visibility_Scene];
				if (ImGui::MenuItem(ICON_FA_COMMENT" Log", 0, visibility_flags[Visibility_Log]))
					visibility_flags[Visibility_Log] = !visibility_flags[Visibility_Log];
				if (ImGui::MenuItem(ICON_FA_TERMINAL" Console", 0, visibility_flags[Visibility_Console]))
					visibility_flags[Visibility_Console] = !visibility_flags[Visibility_Console];
				if (ImGui::MenuItem(ICON_FA_GEAR" Options", 0, visibility_flags[Visibility_Options]))
					visibility_flags[Visibility_Options] = !visibility_flags[Visibility_Options];
				if (ImGui::MenuItem(ICON_FA_BUG" Debug", 0, visibility_flags[Visibility_Debug]))
					visibility_flags[Visibility_Debug] = !visibility_flags[Visibility_Debug];
				if (ImGui::MenuItem(ICON_FA_CLOCK" Stats", 0, visibility_flags[Visibility_Stats]))
					visibility_flags[Visibility_Stats] = !visibility_flags[Visibility_Stats];
				if (ImGui::MenuItem(ICON_FA_CAMERA" Camera", 0, visibility_flags[Visibility_Camera]))
					visibility_flags[Visibility_Camera] = !visibility_flags[Visibility_Camera];
				if (ImGui::MenuItem(ICON_FA_LIGHTBULB" Lights", 0, visibility_flags[Visibility_Lights]))
					visibility_flags[Visibility_Lights] = !visibility_flags[Visibility_Lights];
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu(" Help"))
			{
				ImGui::Text("Help");
				ImGui::Spacing();
				ImGui::EndMenu();
			}
			ImGui::EndMainMenuBar();
		}

		SceneWindow();
		LogWindow();
		ConsoleWindow();
		StatsWindow();
		OptionsWindow();
		DebugWindow();
		CameraWindow();
		LightsWindow();
	}

	void Editor::EndGUI()
	{
		ImGui::Render();
		ImGui_ImplSDLRenderer_RenderDrawData(ImGui::GetDrawData());
		SDLCheck(SDL_SetRenderTarget(sdl_renderer.get(), nullptr));
		SDLCheck(SDL_RenderCopy(sdl_renderer.get(), gui_target.get(), nullptr, nullptr));
	}

	void Editor::SceneWindow()
	{
		if (!visibility_flags[Visibility_Scene]) return;
		ImGui::Begin(ICON_FA_GLOBE" Scene", nullptr, ImGuiWindowFlags_MenuBar);

		if (ImGui::BeginMenuBar())
		{
			if (ImGui::BeginMenu("Path Tracer Output"))
			{
				PathTracerOutput current_output = path_tracer.GetOutput();
				auto AddMenuItem = [&](PathTracerOutput output, Char const* item_name)
					{
						if (ImGui::MenuItem(item_name, nullptr, output == current_output)) { path_tracer.SetOutput(output); }
					};

#define AddPathTracerOutputMenuItem(name) AddMenuItem(PathTracerOutput::##name, #name)
				AddPathTracerOutputMenuItem(Final);
				AddPathTracerOutputMenuItem(Albedo);
				AddPathTracerOutputMenuItem(Normal);
				AddPathTracerOutputMenuItem(Custom);
#undef AddPathTracerOutputMenuItem
				ImGui::EndMenu();
			}
			ImGui::EndMenuBar();
		}

		scene_focused = ImGui::IsWindowFocused();
		ImVec2 v_min = ImGui::GetWindowContentRegionMin();
		ImVec2 v_max = ImGui::GetWindowContentRegionMax();
		v_min.x += ImGui::GetWindowPos().x;
		v_min.y += ImGui::GetWindowPos().y;
		v_max.x += ImGui::GetWindowPos().x;
		v_max.y += ImGui::GetWindowPos().y;
		ImVec2 size(v_max.x - v_min.x, v_max.y - v_min.y);
		ImGui::Image(render_target.get(), size);
		ImGui::End();
	}

	void Editor::LogWindow()
	{
		if (!visibility_flags[Visibility_Log]) return;

		if(editor_sink) editor_sink->Draw(ICON_FA_COMMENT" Log", &visibility_flags[Visibility_Log]);
	}

	void Editor::ConsoleWindow()
	{
		if (!visibility_flags[Visibility_Console]) return;

		editor_console->Draw(ICON_FA_TERMINAL" Console ", &visibility_flags[Visibility_Console]);
	}

	void Editor::StatsWindow()
	{
		if (!visibility_flags[Visibility_Stats]) return;

		if(ImGui::Begin(ICON_FA_CLOCK" Stats", &visibility_flags[Visibility_Stats]))
		{
			ImGuiIO& io = ImGui::GetIO();
			ImGui::Text("FPS: %.1f ms", io.Framerate);
			ImGui::Text("Frame time: %.2f ms", 1000.0f / io.Framerate);

			path_tracer.MemoryUsageGUI();
		}
		ImGui::End();
	}

	void Editor::OptionsWindow()
	{
		if (!visibility_flags[Visibility_Options]) return;

		if(ImGui::Begin(ICON_FA_GEAR" Options", &visibility_flags[Visibility_Options]))
		{
			path_tracer.OptionsGUI();
		}
		ImGui::End();
	}

	void Editor::DebugWindow()
	{
		if (!visibility_flags[Visibility_Debug]) return;

		if(ImGui::Begin(ICON_FA_BUG" Debug", &visibility_flags[Visibility_Debug]))
		{
			if (ImGui::TreeNode("Debug Options"))
			{
				static Char ss_name[32] = {};
				ImGui::InputText("Name", ss_name, sizeof(ss_name) - 1);
				if (ImGui::Button("Take Screenshot"))
				{
					path_tracer.WriteFramebuffer(ss_name);
				}
				ImGui::TreePop();
			}
		}
		ImGui::End();
	}

	void Editor::CameraWindow()
	{
		if (!visibility_flags[Visibility_Camera]) return;

		if(ImGui::Begin(ICON_FA_CAMERA" Camera", &visibility_flags[Visibility_Camera]))
		{
			Vector3 camera_eye = camera.GetPosition();
			ImGui::InputFloat3("Camera Position", &camera_eye.x);
			camera.SetPosition(camera_eye);

			Vector3 camera_look_dir = camera.GetLookDir();
			ImGui::Text("Camera Look Direction: (%f, %f, %f)", camera_look_dir.x, camera_look_dir.y, camera_look_dir.z);
		}
		ImGui::End();
	}

	void Editor::LightsWindow()
	{
		if (!visibility_flags[Visibility_Lights]) return;

		if (ImGui::Begin(ICON_FA_LIGHTBULB " Lights", &visibility_flags[Visibility_Lights]))
		{
			path_tracer.LightsGUI();
		}
		ImGui::End();
	}

}

