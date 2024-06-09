#include "Editor.h"
#include "EditorConsole.h"
#include "EditorSink.h"
#include "Core/Window.h"
#include "Core/Input.h"
#include "Core/Logger.h"
#include "Core/Paths.h"
#include "Scene/Camera.h"
#include "Optix/OptixRenderer.h"
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_sdl.h"
#include "ImGui/imgui_impl_sdlrenderer.h"
#include "FontAwesome/IconsFontAwesome6.h"

namespace amber
{

	Editor::Editor(Window& window, Camera& camera, OptixRenderer& renderer)
		: window(window), camera(camera), renderer(renderer)
	{
		SDLCheck(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0);
		window.GetWindowEvent().AddMember(&Editor::OnWindowEvent, *this);
		g_Input.GetInputEvents().key_pressed.AddMember(&Editor::OnKeyPressed, *this);
		g_Input.GetInputEvents().window_resized_event.AddMember(&Editor::OnResize, *this);

		editor_console = std::make_unique<EditorConsole>();

		uint32 renderer_flags = SDL_RENDERER_ACCELERATED;
		renderer_flags |= SDL_RENDERER_PRESENTVSYNC;
		sdl_renderer.reset(SDL_CreateRenderer(window.sdl_window.get(), -1, renderer_flags));
		SDLCheck(sdl_renderer.get());

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
		io.ConfigWindowsResizeFromEdges = true;
		ImGui::StyleColorsDark();
		ImGui_ImplSDL2_InitForSDLRenderer(window.sdl_window.get(), sdl_renderer.get());
		ImGui_ImplSDLRenderer_Init(sdl_renderer.get());

		ImFontConfig font_config{};
		std::string font_name = paths::FontsDir() + "roboto/Roboto-Light.ttf";
		io.Fonts->AddFontFromFileTTF(font_name.c_str(), 16.0f, &font_config);
		font_config.MergeMode = true;
		ImWchar const icon_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
		std::string icon_name = paths::FontsDir() + "FontAwesome/" FONT_ICON_FILE_NAME_FAS;
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
		float dt = ImGui::GetIO().DeltaTime;

		camera.Tick(dt);
		renderer.Update(dt);
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

	void Editor::OnResize(int32 w, int32 h)
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

		renderer.OnResize(w, h);
		camera.SetAspectRatio((float)w / h);
	}

	void Editor::OnWindowEvent(WindowEventData const& data)
	{
		g_Input.OnWindowEvent(data);
		if (gui_enabled) ImGui_ImplSDL2_ProcessEvent(data.event);
	}

	void Editor::SetStyle()
	{
		ImGuiStyle& style = ImGui::GetStyle();
		ImGui::StyleColorsDark(&style);

		style.FrameRounding = 0.0f;
		style.GrabRounding = 1.0f;
		style.WindowRounding = 0.0f;
		style.IndentSpacing = 10.0f;
		style.WindowPadding = ImVec2(5, 5);
		style.FramePadding = ImVec2(2, 2);
		style.WindowBorderSize = 1.00f;
		style.ChildBorderSize = 1.00f;
		style.PopupBorderSize = 1.00f;
		style.FrameBorderSize = 1.00f;
		style.ScrollbarSize = 20.0f;
		style.WindowMenuButtonPosition = ImGuiDir_Right;

		ImVec4* colors = ImGui::GetStyle().Colors;
		colors[ImGuiCol_Text] = ImVec4(0.85f, 0.87f, 0.91f, 0.88f);
		colors[ImGuiCol_TextDisabled] = ImVec4(0.49f, 0.50f, 0.53f, 1.00f);
		colors[ImGuiCol_WindowBg] = ImVec4(0.18f, 0.20f, 0.25f, 1.00f);
		colors[ImGuiCol_ChildBg] = ImVec4(0.16f, 0.17f, 0.20f, 1.00f);
		colors[ImGuiCol_PopupBg] = ImVec4(0.23f, 0.26f, 0.32f, 1.00f);
		colors[ImGuiCol_Border] = ImVec4(0.14f, 0.16f, 0.19f, 1.00f);
		colors[ImGuiCol_BorderShadow] = ImVec4(0.09f, 0.09f, 0.09f, 0.00f);
		colors[ImGuiCol_FrameBg] = ImVec4(0.23f, 0.26f, 0.32f, 1.00f);
		colors[ImGuiCol_FrameBgHovered] = ImVec4(0.56f, 0.74f, 0.73f, 1.00f);
		colors[ImGuiCol_FrameBgActive] = ImVec4(0.53f, 0.75f, 0.82f, 1.00f);
		colors[ImGuiCol_TitleBg] = ImVec4(0.16f, 0.16f, 0.20f, 1.00f);
		colors[ImGuiCol_TitleBgActive] = ImVec4(0.16f, 0.16f, 0.20f, 1.00f);
		colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.16f, 0.16f, 0.20f, 1.00f);
		colors[ImGuiCol_MenuBarBg] = ImVec4(0.16f, 0.16f, 0.20f, 1.00f);
		colors[ImGuiCol_ScrollbarBg] = ImVec4(0.18f, 0.20f, 0.25f, 1.00f);
		colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.23f, 0.26f, 0.32f, 0.60f);
		colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.23f, 0.26f, 0.32f, 1.00f);
		colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.23f, 0.26f, 0.32f, 1.00f);
		colors[ImGuiCol_CheckMark] = ImVec4(0.37f, 0.51f, 0.67f, 1.00f);
		colors[ImGuiCol_SliderGrab] = ImVec4(0.51f, 0.63f, 0.76f, 1.00f);
		colors[ImGuiCol_SliderGrabActive] = ImVec4(0.37f, 0.51f, 0.67f, 1.00f);
		colors[ImGuiCol_Button] = ImVec4(0.18f, 0.20f, 0.25f, 1.00f);
		colors[ImGuiCol_ButtonHovered] = ImVec4(0.51f, 0.63f, 0.76f, 1.00f);
		colors[ImGuiCol_ButtonActive] = ImVec4(0.37f, 0.51f, 0.67f, 1.00f);
		colors[ImGuiCol_Header] = ImVec4(0.51f, 0.63f, 0.76f, 1.00f);
		colors[ImGuiCol_HeaderHovered] = ImVec4(0.53f, 0.75f, 0.82f, 1.00f);
		colors[ImGuiCol_HeaderActive] = ImVec4(0.37f, 0.51f, 0.67f, 1.00f);
		colors[ImGuiCol_SeparatorHovered] = ImVec4(0.56f, 0.74f, 0.73f, 1.00f);
		colors[ImGuiCol_SeparatorActive] = ImVec4(0.53f, 0.75f, 0.82f, 1.00f);
		colors[ImGuiCol_ResizeGrip] = ImVec4(0.53f, 0.75f, 0.82f, 0.86f);
		colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.61f, 0.74f, 0.87f, 1.00f);
		colors[ImGuiCol_ResizeGripActive] = ImVec4(0.37f, 0.51f, 0.67f, 1.00f);
		colors[ImGuiCol_Tab] = ImVec4(0.18f, 0.20f, 0.25f, 1.00f);
		colors[ImGuiCol_TabHovered] = ImVec4(0.22f, 0.24f, 0.31f, 1.00f);
		colors[ImGuiCol_TabActive] = ImVec4(0.23f, 0.26f, 0.32f, 1.00f);
		colors[ImGuiCol_TabUnfocused] = ImVec4(0.13f, 0.15f, 0.18f, 1.00f);
		colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.17f, 0.19f, 0.23f, 1.00f);
		colors[ImGuiCol_PlotHistogram] = ImVec4(0.56f, 0.74f, 0.73f, 1.00f);
		colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.53f, 0.75f, 0.82f, 1.00f);
		colors[ImGuiCol_TextSelectedBg] = ImVec4(0.37f, 0.51f, 0.67f, 1.00f);
		colors[ImGuiCol_NavHighlight] = ImVec4(0.53f, 0.75f, 0.82f, 0.86f);
	}

	void Editor::OnKeyPressed(KeyCode keycode)
	{
		if (keycode == KeyCode::I)
		{
			gui_enabled = !gui_enabled;
			g_Input.SetMouseVisibility(gui_enabled);
		}
	}

	void Editor::Begin()
	{
		SDL_SetRenderDrawColor(sdl_renderer.get(), 0, 0, 0, SDL_ALPHA_OPAQUE);
		SDL_RenderClear(sdl_renderer.get());
	}

	void Editor::Render()
	{
		renderer.Render(camera, sample_count);
		auto const& fb = renderer.GetFramebuffer();

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
		ImGui::Begin(ICON_FA_GLOBE"Scene");
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

	void Editor::GUI()
	{
		if (ImGui::BeginMainMenuBar())
		{
			if (ImGui::BeginMenu(ICON_FA_WINDOW_MAXIMIZE" Windows"))
			{
				if (ImGui::MenuItem(ICON_FA_COMMENT" Log", 0, visibility_flags[Visibility_Log]))
					visibility_flags[Visibility_Log] = !visibility_flags[Visibility_Log];
				if (ImGui::MenuItem(ICON_FA_TERMINAL" Console", 0, visibility_flags[Visibility_Console]))
					visibility_flags[Visibility_Console] = !visibility_flags[Visibility_Console];
				if (ImGui::MenuItem(ICON_FA_GEAR" Settings", 0, visibility_flags[Visibility_Settings]))
					visibility_flags[Visibility_Settings] = !visibility_flags[Visibility_Settings];
				if (ImGui::MenuItem(ICON_FA_CLOCK" Stats", 0, visibility_flags[Visibility_Stats]))
					visibility_flags[Visibility_Stats] = !visibility_flags[Visibility_Stats];
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
		LogWindow();
		ConsoleWindow();
		StatsWindow();
		SettingsWindow();
	}

	void Editor::EndGUI()
	{
		ImGui::Render();
		ImGui_ImplSDLRenderer_RenderDrawData(ImGui::GetDrawData());
		SDLCheck(SDL_SetRenderTarget(sdl_renderer.get(), nullptr));
		SDLCheck(SDL_RenderCopy(sdl_renderer.get(), gui_target.get(), nullptr, nullptr));
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
		ImGui::Begin(ICON_FA_CLOCK" Stats");
		{
			ImGuiIO& io = ImGui::GetIO();
			ImGui::Text("FPS: %.1f ms", io.Framerate);
			ImGui::Text("Frame time: %.2f ms", 1000.0f / io.Framerate);
		}
		ImGui::End();
	}

	void Editor::SettingsWindow()
	{
		if (!visibility_flags[Visibility_Settings]) return;
		ImGui::Begin(ICON_FA_GEAR" Settings");
		{
			ImGuiIO& io = ImGui::GetIO();
			ImGui::SliderInt("Samples", &sample_count, 1, 64);
			ImGui::SliderInt("Max Depth", &max_depth, 1, renderer.GetMaxDepth());
		}
		ImGui::End();
	}

}

