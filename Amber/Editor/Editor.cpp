#include "Editor.h"
#include "EditorConsole.h"
#include "EditorSink.h"
#include "Platform/Window.h"
#include "Platform/Input.h"
#include "Core/Log.h"
#include "Core/Paths.h"
#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include "Device/PathTracer.h"
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_sdl2.h"
#include "ImGui/imgui_impl_sdlrenderer2.h"
#include "FontAwesome/IconsFontAwesome6.h"

namespace amber
{
	Editor::Editor(Window& window, Camera& camera, PathTracerBase& path_tracer)
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
		ini_file = paths::IniDir + "imgui.ini";
		io.IniFilename = ini_file.c_str();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
		io.ConfigWindowsResizeFromEdges = true;
		ImGui::StyleColorsDark();
		ImGui_ImplSDL2_InitForSDLRenderer(window.sdl_window.get(), sdl_renderer.get());
		ImGui_ImplSDLRenderer2_Init(sdl_renderer.get());

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
		ImGui_ImplSDLRenderer2_Shutdown();
		ImGui_ImplSDL2_Shutdown();
		ImGui::DestroyContext();
		SDL_Quit();
	}

	void Editor::Run()
	{
		g_Input.Tick();
		if (gui_enabled) 
		{
			camera.Enable(scene_focused);
		}
		else 
		{
			camera.Enable(true);
		}

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
		if (gui_enabled) 
		{
			ImGui_ImplSDL2_ProcessEvent(data.event);
		}
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

		for (Int i = 0; i <= ImGuiCol_COUNT; i++)
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

		Int width, height, pitch = -1; void* data = nullptr;
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
		ImGui_ImplSDLRenderer2_NewFrame();
		ImGui_ImplSDL2_NewFrame();
		ImGui::NewFrame();

		SDLCheck(SDL_SetRenderTarget(sdl_renderer.get(), gui_target.get()));
		SDL_RenderClear(sdl_renderer.get());
		ImGuiID dockspace_id = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);
	}

	void Editor::GUI()
	{
		if (ImGui::BeginMainMenuBar())
		{
			if (ImGui::BeginMenu(ICON_FA_WINDOW_MAXIMIZE" Windows"))
			{
				if (ImGui::MenuItem(ICON_FA_GLOBE" Scene",       0, visibility_flags[Visibility_Scene]))
					visibility_flags[Visibility_Scene]      = !visibility_flags[Visibility_Scene];
				if (ImGui::MenuItem(ICON_FA_SLIDERS" Properties", 0, visibility_flags[Visibility_Properties]))
					visibility_flags[Visibility_Properties] = !visibility_flags[Visibility_Properties];
				if (ImGui::MenuItem(ICON_FA_CLOCK" Stats",        0, visibility_flags[Visibility_Stats]))
					visibility_flags[Visibility_Stats]      = !visibility_flags[Visibility_Stats];
				if (ImGui::MenuItem(ICON_FA_BUG" Debug",          0, visibility_flags[Visibility_Debug]))
					visibility_flags[Visibility_Debug]      = !visibility_flags[Visibility_Debug];
				if (ImGui::MenuItem(ICON_FA_TERMINAL" Console",   0, visibility_flags[Visibility_Console]))
					visibility_flags[Visibility_Console]    = !visibility_flags[Visibility_Console];
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
		PropertiesWindow();
		StatsWindow();
		DebugWindow();
		ConsoleWindow();
	}

	void Editor::EndGUI()
	{
		ImGui::Render();
		ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), sdl_renderer.get());
		SDLCheck(SDL_SetRenderTarget(sdl_renderer.get(), nullptr));
		SDLCheck(SDL_RenderCopy(sdl_renderer.get(), gui_target.get(), nullptr, nullptr));
	}

	void Editor::SceneWindow()
	{
		if (!visibility_flags[Visibility_Scene]) 
		{
			return;
		}
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

#define AddPathTracerOutputMenuItem(name) AddMenuItem(PathTracerOutput::name, #name)
				AddPathTracerOutputMenuItem(Final);
				AddPathTracerOutputMenuItem(Albedo);
				AddPathTracerOutputMenuItem(Normal);
				AddPathTracerOutputMenuItem(UV);
				AddPathTracerOutputMenuItem(MaterialID);
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

	void Editor::PropertiesWindow()
	{
		if (!visibility_flags[Visibility_Properties]) return;
		if (!ImGui::Begin(ICON_FA_SLIDERS" Properties", &visibility_flags[Visibility_Properties]))
		{
			ImGui::End();
			return;
		}

		if (ImGui::CollapsingHeader(ICON_FA_MICROCHIP" Renderer", ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::Text("Backend: %s", GetBackendName(path_tracer.GetBackend()).c_str());

			Uint tri_count = path_tracer.GetTriangleCount();
			if (tri_count > 0) 
			{
				ImGui::Text("Triangles: %u", tri_count);
			}

			if (path_tracer.SupportsAccumulation())
			{
				Bool acc = path_tracer.GetAccumulate();
				if (ImGui::Checkbox("Accumulate", &acc))
				{
					path_tracer.SetAccumulate(acc);
				}
			}

			if (path_tracer.GetSampleCount() > 0)
			{
				Int samples = path_tracer.GetSampleCount();
				if (ImGui::SliderInt("Samples Per Pixel", &samples, 1, 128))
				{
					path_tracer.SetSampleCount(samples);
				}
			}

			{
				Int depth = path_tracer.GetDepthCount();
				if (ImGui::SliderInt("Max Depth", &depth, 1, path_tracer.GetMaxDepth()))
				{
					path_tracer.SetDepthCount(depth);
				}
			}
		}

		if (path_tracer.HasDenoiser() && ImGui::CollapsingHeader(ICON_FA_WAND_MAGIC_SPARKLES" Denoiser"))
		{
			path_tracer.DenoiserGUI();
		}

		if (ImGui::CollapsingHeader(ICON_FA_VIDEO" Camera"))
		{
			Vector3 camera_eye = camera.GetPosition();
			if (ImGui::InputFloat3("Position", &camera_eye.x))
			{
				camera.SetPosition(camera_eye);
			}

			Vector3 look_dir = camera.GetLookDir();
			ImGui::Text("Look Dir: (%.2f, %.2f, %.2f)", look_dir.x, look_dir.y, look_dir.z);
			ImGui::Text("FoV: %.1f", camera.GetFovY());
		}

		if (ImGui::CollapsingHeader(ICON_FA_LIGHTBULB" Lights"))
		{
			ImGui::Text("Light count: %zu", path_tracer.GetScene().lights.size());
			if (path_tracer.HasLightEditor())
			{
				ImGui::Separator();
				path_tracer.LightEditorGUI();
			}
		}

		if (ImGui::CollapsingHeader(ICON_FA_CLOUD_SUN" Environment"))
		{
			Scene const& scene = path_tracer.GetScene();
			if (scene.environment)
			{
				ImGui::Text("Resolution: %d x %d", scene.environment->GetWidth(), scene.environment->GetHeight());
				ImGui::Text("Format: %s", scene.environment->IsHDR() ? "HDR (RGBA32F)" : (scene.environment->IsSRGB() ? "LDR sRGB" : "LDR"));
			}
			else
			{
				ImGui::TextDisabled("No environment map loaded");
			}
		}

		if (path_tracer.HasPostProcessing() && ImGui::CollapsingHeader(ICON_FA_FILM" Post Processing"))
		{
			path_tracer.PostProcessingGUI();
		}

		ImGui::End();
	}

	void Editor::DebugWindow()
	{
		if (!visibility_flags[Visibility_Debug]) return;
		if (!ImGui::Begin(ICON_FA_BUG" Debug", &visibility_flags[Visibility_Debug]))
		{
			ImGui::End();
			return;
		}

		if (ImGui::CollapsingHeader("Screenshot", ImGuiTreeNodeFlags_DefaultOpen))
		{
			static Char ss_name[64] = "screenshot";
			ImGui::SetNextItemWidth(-1);
			ImGui::InputText("##ScreenshotName", ss_name, sizeof(ss_name) - 1);
			if (ImGui::Button(ICON_FA_CAMERA" Take Screenshot", ImVec2(-1, 0)))
			{
				path_tracer.WriteFramebuffer(ss_name);
			}
		}

		if (path_tracer.HasBVHDebug() && ImGui::CollapsingHeader("BVH", ImGuiTreeNodeFlags_DefaultOpen))
		{
			path_tracer.BVHDebugGUI();
		}

		ImGui::End();
	}

	void Editor::StatsWindow()
	{
		if (!visibility_flags[Visibility_Stats]) return;
		if (!ImGui::Begin(ICON_FA_CLOCK" Stats", &visibility_flags[Visibility_Stats]))
		{
			ImGui::End();
			return;
		}

		ImGuiIO& io = ImGui::GetIO();
		ImGui::Text("FPS:         %.1f", io.Framerate);
		ImGui::Text("Frame time:  %.2f ms", 1000.0f / io.Framerate);
		ImGui::Text("Frame:       %u", path_tracer.GetFrameIndex());

		Float render_time = path_tracer.GetRenderTime();
		if (render_time > 0.0f)
		{
			ImGui::Text("Render time: %.2f ms", render_time);
		}

		auto const& fb = path_tracer.GetFramebuffer();
		ImGui::Text("Resolution:  %llu x %llu",
			static_cast<unsigned long long>(fb.Cols()),
			static_cast<unsigned long long>(fb.Rows()));

		Uint tri_count = path_tracer.GetTriangleCount();
		if (tri_count > 0)
		{
			ImGui::Text("Triangles:   %u", tri_count);
		}

		ImGui::Separator();

		Uint64 mem = path_tracer.GetMemoryUsage();
		if (mem > 0)
		{
			ImGui::Text("Memory:      %.2f MB", mem / (1024.0 * 1024.0));
		}
		else
		{
			ImGui::TextDisabled("Memory:      N/A");
		}

		ImGui::End();
	}

	void Editor::ConsoleWindow()
	{
		if (!visibility_flags[Visibility_Console])
		{
			return;
		}

		if (editor_sink) 
		{
			editor_sink->Draw(ICON_FA_COMMENT" Log", &visibility_flags[Visibility_Console]);
		}
		editor_console->Draw(ICON_FA_TERMINAL" Console", &visibility_flags[Visibility_Console]);
	}

}
