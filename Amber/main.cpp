#define SDL_MAIN_HANDLED
#include "Utilities/SDLUtil.h"
#include "CLI/CLI.hpp"

#include "Core/Window.h"
#include "Core/Logger.h"
#include "Core/Paths.h"
#include "Editor/Editor.h"
#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include "Optix/OptixRenderer.h"
#include "Utilities/CpuBuffer2D.h"
#include "Utilities/JsonUtil.h"


using namespace amber;

struct Config
{
	std::string scene_file;
	float  scene_scale;
	uint32 width;
	uint32 height;
	uint32 max_depth;
	uint32 samples_per_pixel;
	Camera camera;
};
bool ParseConfig(char const* config_file, Config& cfg);


int main(int argc, char* argv[])
{
	std::string config_file, log_file;
	bool use_editor = false, maximize_window = false, stats_enabled = false;
	{
		CLI::App cli_parser{ "Lavender" };
		cli_parser.add_option("--config-file", config_file, "Config file");
		cli_parser.add_option("--log-file", log_file, "Log file");
		CLI::Option* use_editor_opt = cli_parser.add_flag("--editor", "Use editor");
		CLI::Option* max_window_opt = cli_parser.add_flag("--max", "Maximize editor window");
		CLI11_PARSE(cli_parser, argc, argv);
		if (log_file.empty()) log_file = "lavender.log";
		if (config_file.empty()) config_file = "config.json";
		use_editor = (bool)*use_editor_opt;
		maximize_window = (bool)*max_window_opt;
	}
	g_LogManager.Initialize(log_file.c_str(), LogLevel::Debug);

	Config cfg{};
	bool config_parsed = ParseConfig(config_file.c_str(), cfg);
	if (!config_parsed)
	{
		return EXIT_FAILURE;
	}
	
	std::unique_ptr<Scene> scene = nullptr;
	try
	{
		scene = LoadScene(cfg.scene_file.c_str(), cfg.scene_scale);
		if (!scene)
		{
			LAV_ERROR("Scene loading failed!");
			return EXIT_FAILURE;
		}
	}
	catch (std::runtime_error const& e)
	{
		LAV_ERROR("{}", e.what());
		return EXIT_FAILURE;
	}

	Camera camera = std::move(cfg.camera);
	OptixRenderer renderer(cfg.width, cfg.height, std::move(scene));
	if(use_editor)
	{
		Window window(cfg.width, cfg.height, "lavender");
		if (maximize_window) window.Maximize();
		Editor editor(window, camera, renderer);
		editor.SetEditorSink(g_LogManager.GetEditorSink());
		editor.SetDefaultRendererOptions(cfg.samples_per_pixel, cfg.max_depth);
		while (window.Loop())
		{
			editor.Run();
		}
	}
	else
	{
		renderer.Render(camera, cfg.samples_per_pixel);
		renderer.WriteFramebuffer("test.png"); 
	}
	g_LogManager.Destroy();

	return 0;
}

bool ParseConfig(char const* config_file, Config& cfg)
{
	json json_scene;
	try
	{
		JsonParams scene_params = json::parse(std::ifstream(paths::ConfigDir() + config_file));
		json_scene = scene_params.FindJson("scene");
	}
	catch (json::parse_error const& e)
	{
		LAV_ERROR("JSON parsing error: {}! ", e.what());
		return false;
	}
	JsonParams scene_params(json_scene);

	std::string scene_file;
	bool scene_file_found = scene_params.Find<std::string>("scene file", scene_file);
	if (!scene_file_found)
	{
		LAV_ERROR("Scene file not specified in config file!");
		return false;
	}

	cfg.scene_file = paths::SceneDir() + scene_file;
	cfg.scene_scale = scene_params.FindOr<float>("scene scale", 1.0f);
	cfg.width = scene_params.FindOr<uint32>("width", 1080);
	cfg.height = scene_params.FindOr<uint32>("height", 720);
	cfg.max_depth = scene_params.FindOr<uint32>("max depth", 4);
	cfg.samples_per_pixel = scene_params.FindOr<uint32>("samples per pixel", 16);

	json camera_json = scene_params.FindJson("camera");
	if (camera_json.is_null())
	{
		LAV_ERROR("Missing camera parameters in config file!");
		return false;
	}

	JsonParams camera_params(camera_json);

	float camera_position[3] = { 0.0f, 0.0f, 0.0f };
	camera_params.FindArray("position", camera_position);
	cfg.camera.SetEye(Vector3(camera_position));

	float camera_up[3] = { 0.0f, 1.0f, 0.0f };
	camera_params.FindArray("up", camera_up);
	cfg.camera.SetUp(Vector3(camera_up));

	float look_at[3] = { 0.0f, 0.0f, 1.0f };
	camera_params.FindArray("look_at", look_at);
	cfg.camera.SetLookat(Vector3(look_at));

	float fovy = camera_params.FindOr<float>("fov", 45.0f);
	cfg.camera.SetFovY(fovy);
	cfg.camera.SetAspectRatio(cfg.width * 1.0f / cfg.height);
	return true;
}
