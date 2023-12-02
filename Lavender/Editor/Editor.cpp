#include "Editor.h"
#include "Core/Window.h"
#include "Core/Input.h"
#include "Core/Logger.h"
#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_sdlrenderer.h"

namespace lavender
{

	Editor::Editor(Window& window) : window(window)
	{
		window.GetWindowEvent().AddMember(&Editor::OnWindowEvent, *this);
		window.GetWindowEvent().AddMember(&Input::OnWindowEvent, g_Input);

		uint32 renderer_flags = SDL_RENDERER_ACCELERATED;
		renderer_flags |= SDL_RENDERER_PRESENTVSYNC;
		renderer.reset(SDL_CreateRenderer(window.sdl_window.get(), -1, renderer_flags));
		SDLCheck(renderer.get());

		SDL_RendererInfo renderer_info;
		SDLCheck(SDL_GetRendererInfo(renderer.get(), &renderer_info));
		LAVENDER_INFO("Renderer name: {}", renderer_info.name);

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
		io.ConfigWindowsResizeFromEdges = true;
		ImGui::StyleColorsDark();
		ImGui_ImplSDL2_InitForSDLRenderer(window.sdl_window.get(), renderer.get());
		ImGui_ImplSDLRenderer_Init(renderer.get());

		render_target.reset(SDL_CreateTexture(renderer.get(),
			SDL_PIXELFORMAT_RGBA32,
			SDL_TEXTUREACCESS_STREAMING,
			window.Width(), window.Height()));
		SDLCheck(render_target.get());

		gui_target.reset(SDL_CreateTexture(
						 renderer.get(), 
						 SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET,
						 window.Width(), window.Height()));
		SDLCheck(gui_target.get());
		SetStyle();
	}

	Editor::~Editor()
	{
		ImGui_ImplSDLRenderer_Shutdown();
		ImGui_ImplSDL2_Shutdown();
		ImGui::DestroyContext();
	}

	void Editor::Run()
	{
		g_Input.Tick();
		Update();
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
				SDLCheck(SDL_RenderCopy(renderer.get(), render_target.get(), nullptr, nullptr));
			}
		}
		End();
	}

	void Editor::OnResize(int32 w, int32 h)
	{
		render_target.reset(SDL_CreateTexture(renderer.get(),
			SDL_PIXELFORMAT_RGBA32,
			SDL_TEXTUREACCESS_STREAMING,
			window.Width(), window.Height()));
		SDLCheck(render_target.get());

		gui_target.reset(SDL_CreateTexture(renderer.get(),
			SDL_PIXELFORMAT_RGBA8888,
			SDL_TEXTUREACCESS_TARGET,
			w, h));
		SDLCheck(gui_target.get());
	}

	void Editor::OnWindowEvent(WindowEventData const& data)
	{
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

	void Editor::Update()
	{
		if (g_Input.GetKey(KeyCode::I))
		{
			gui_enabled = !gui_enabled;
			g_Input.SetMouseVisibility(gui_enabled);
		}
	}

	void Editor::Begin()
	{
		SDL_SetRenderDrawColor(renderer.get(), 0, 0, 0, SDL_ALPHA_OPAQUE);
		SDL_RenderClear(renderer.get());
	}

	void Editor::Render()
	{

	}

	void Editor::End()
	{
		SDL_RenderPresent(renderer.get());
	}

	void Editor::BeginGUI()
	{
		ImGui_ImplSDLRenderer_NewFrame();
		ImGui_ImplSDL2_NewFrame();
		ImGui::NewFrame();

		SDLCheck(SDL_SetRenderTarget(renderer.get(), gui_target.get()));
		SDL_RenderClear(renderer.get());
		
		ImGuiID dockspace_id = ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);
		ImGui::Begin("Lavender");
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

	}

	void Editor::EndGUI()
	{
		ImGui::Render();
		ImGui_ImplSDLRenderer_RenderDrawData(ImGui::GetDrawData());
		SDLCheck(SDL_SetRenderTarget(renderer.get(), nullptr));
		SDLCheck(SDL_RenderCopy(renderer.get(), gui_target.get(), nullptr, nullptr));
	}

}
