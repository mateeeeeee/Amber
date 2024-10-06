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
	std::string model_file;
	float  model_scale;
	std::string scene_environment;
	uint32 width;
	uint32 height;
	uint32 max_depth;
	uint32 samples_per_pixel;
	Camera camera;
};
bool ParseSceneConfig(char const* config_file, Config& cfg);


int main(int argc, char* argv[])
{
	std::string config_file, log_file;
	bool use_editor = true, maximize_window = false, stats_enabled = false;
	{
		CLI::App cli_parser{ "Lavender" };
		cli_parser.add_option("--config-file", config_file, "Config file");
		cli_parser.add_option("--log-file", log_file, "Log file");
		CLI::Option* no_editor_opt = cli_parser.add_flag("--noeditor", "Don't use editor");
		CLI::Option* max_window_opt = cli_parser.add_flag("--max", "Maximize editor window");
		CLI11_PARSE(cli_parser, argc, argv);
		if (log_file.empty()) log_file = "lavender.log";
		if (config_file.empty()) config_file = "sanmiguel.json";
		use_editor = !(bool)*no_editor_opt;
		maximize_window = (bool)*max_window_opt;
	}
	g_LogManager.Initialize(log_file.c_str(), LogLevel::Debug);

	Config cfg{};
	if (!ParseSceneConfig(config_file.c_str(), cfg))
	{
		AMBER_ERROR("Config parsing failed!");
		return EXIT_FAILURE;
	}
	
	std::unique_ptr<Scene> scene = nullptr;
	try
	{
		scene = LoadScene(cfg.model_file.c_str(), cfg.scene_environment.c_str(), cfg.model_scale);
		if (!scene)
		{
			AMBER_ERROR("Scene loading failed!");
			return EXIT_FAILURE;
		}
	}
	catch (std::runtime_error const& e)
	{
		AMBER_ERROR("{}", e.what());
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
		editor.SetDefaultOptions(cfg.samples_per_pixel, cfg.max_depth);
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

bool ParseSceneConfig(char const* scene_config, Config& cfg)
{
	json json_scene;
	try
	{
		JsonParams scene_params = json::parse(std::ifstream(paths::SceneDir + scene_config));
		json_scene = scene_params.FindJson("scene");
	}
	catch (json::parse_error const& e)
	{
		AMBER_ERROR("JSON parsing error: {}! ", e.what());
		return false;
	}
	JsonParams scene_params(json_scene);

	std::string model_file;
	bool model_file_found = scene_params.Find<std::string>("model file", model_file);
	if (!model_file_found)
	{
		AMBER_ERROR("Scene file not specified in config file!");
		return false;
	}
	std::string scene_environment;
	scene_params.Find<std::string>("scene environment", scene_environment);

	cfg.model_file = paths::ModelDir + model_file;
	cfg.model_scale = scene_params.FindOr<float>("model scale", 1.0f);
	cfg.scene_environment = paths::ModelDir + scene_environment;
	cfg.width = scene_params.FindOr<uint32>("width", 1080);
	cfg.height = scene_params.FindOr<uint32>("height", 720);
	cfg.max_depth = scene_params.FindOr<uint32>("max depth", 4);
	cfg.samples_per_pixel = scene_params.FindOr<uint32>("samples per pixel", 16);

	json camera_json = scene_params.FindJson("camera");
	if (camera_json.is_null())
	{
		AMBER_ERROR("Missing camera parameters in config file!");
		return false;
	}

	JsonParams camera_params(camera_json);

	float camera_position[3] = { 0.0f, 0.0f, 0.0f };
	camera_params.FindArray("position", camera_position);
	float look_at[3] = { 0.0f, 0.0f, 1.0f };
	camera_params.FindArray("look_at", look_at);
	cfg.camera.Initialize(Vector3(camera_position), Vector3(look_at));

	float fovy = camera_params.FindOr<float>("fov", 45.0f);
	cfg.camera.SetFovY(fovy);
	cfg.camera.SetAspectRatio(cfg.width * 1.0f / cfg.height);
	return true;
}
