#include "SDLImGuiManager.h"
#include "Platform/Windows/Window.h"
#include "Core/Paths.h"
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_sdl2.h"
#include "ImGui/imgui_impl_sdlrenderer2.h"
#include "FontAwesome/IconsFontAwesome6.h"

namespace amber
{
	SDLImGuiManager::SDLImGuiManager(Window& window, SDLRendererPtr const& renderer)
		: window(window), renderer(renderer)
	{
	}

	SDLImGuiManager::~SDLImGuiManager()
	{
		Shutdown();
	}

	void SDLImGuiManager::Initialize()
	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		ini_file = paths::IniDir + "imgui.ini";
		io.IniFilename = ini_file.c_str();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
		io.ConfigWindowsResizeFromEdges = true;
		ImGui::StyleColorsDark();
		ImGui_ImplSDL2_InitForSDLRenderer(window.sdl_window.get(), renderer.get());
		ImGui_ImplSDLRenderer2_Init(renderer.get());

		ImFontConfig font_config{};
		std::string font_name = paths::FontsDir + "roboto/Roboto-Light.ttf";
		io.Fonts->AddFontFromFileTTF(font_name.c_str(), 16.0f, &font_config);
		font_config.MergeMode = true;
		ImWchar const icon_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
		std::string icon_name = paths::FontsDir + "FontAwesome/" FONT_ICON_FILE_NAME_FAS;
		io.Fonts->AddFontFromFileTTF(icon_name.c_str(), 15.0f, &font_config, icon_ranges);
		io.Fonts->Build();
	}

	void SDLImGuiManager::Shutdown()
	{
		ImGui_ImplSDLRenderer2_Shutdown();
		ImGui_ImplSDL2_Shutdown();
		ImGui::DestroyContext();
	}

	void SDLImGuiManager::BeginFrame()
	{
		ImGui_ImplSDLRenderer2_NewFrame();
		ImGui_ImplSDL2_NewFrame();
		ImGui::NewFrame();
	}

	void SDLImGuiManager::EndFrame()
	{
		ImGui::Render();
	}

	void SDLImGuiManager::Render()
	{
		ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer.get());
	}
}
