#define SDL_MAIN_HANDLED
#include "Utilities/SDLUtil.h"
#include "CLI/CLI.hpp"

#include "Core/Window.h"
#include "Core/Logger.h"
#include "Core/Paths.h"
#include "Core/ConsoleManager.h"
#include "Editor/Editor.h"
#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include "Optix/OptixRenderer.h"
#include "Utilities/CpuBuffer2D.h"
#include "Utilities/JsonUtil.h"


using namespace amber;

struct SceneConfig
{
	std::string model_file;
	Float  model_scale;
	std::string scene_environment;
	Uint32 width;
	Uint32 height;
	Uint32 max_depth;
	Uint32 samples_per_pixel;
	Camera camera;
};
Bool ParseSceneConfig(Char const* config_file, SceneConfig& cfg);
void ProcessCVarIniFile(Char const* cvar_file);

int main(Sint32 argc, Char* argv[])
{
	std::string config_file, log_file, output_file;
	Bool use_editor = true, maximize_window = false, stats_enabled = false;
	{
		CLI::App cli_parser{ "Amber" };
		cli_parser.add_option("--config-file", config_file, "Config file");
		cli_parser.add_option("--log-file", log_file, "Log file");
		cli_parser.add_option("--output-file", output_file, "Output file");
		CLI::Option* no_editor_opt = cli_parser.add_flag("--noeditor", "Don't use editor");
		CLI::Option* max_window_opt = cli_parser.add_flag("--max", "Maximize editor window");
		CLI11_PARSE(cli_parser, argc, argv);
		if (log_file.empty()) log_file = "amber.log";
		if (config_file.empty()) config_file = "sanmiguel.json";
		if (output_file.empty()) output_file = "output";
		use_editor = !(Bool)*no_editor_opt;
		maximize_window = (Bool)*max_window_opt;
	}
#ifdef _DEBUG
	g_LogManager.Initialize(log_file.c_str(), LogLevel::Debug);
#else 
	g_LogManager.Initialize(log_file.c_str(), LogLevel::Error);
#endif

	SceneConfig cfg{};
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
	ProcessCVarIniFile("cvars.ini");
	if(use_editor)
	{
		Window window(cfg.width, cfg.height, "amber");
		if (maximize_window) window.Maximize();
		Editor editor(window, camera, renderer);
		editor.SetEditorSink(g_LogManager.GetEditorSink());
		renderer.SetDepthCount(cfg.max_depth);
		renderer.SetSampleCount(cfg.samples_per_pixel);
		while (window.Loop())
		{
			editor.Run();
		}
	}
	else
	{
		renderer.SetDepthCount(cfg.max_depth);
		renderer.SetSampleCount(cfg.samples_per_pixel);
		renderer.Render(camera);
		renderer.WriteFramebuffer(output_file.c_str()); 
	}
	g_LogManager.Destroy();

	return 0;
}

Bool ParseSceneConfig(Char const* scene_config, SceneConfig& cfg)
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
	Bool model_file_found = scene_params.Find<std::string>("model file", model_file);
	if (!model_file_found)
	{
		AMBER_ERROR("Scene file not specified in config file!");
		return false;
	}
	std::string scene_environment;
	scene_params.Find<std::string>("scene environment", scene_environment);

	cfg.model_file = paths::ModelDir + model_file;
	cfg.model_scale = scene_params.FindOr<Float>("model scale", 1.0f);
	cfg.scene_environment = paths::ModelDir + scene_environment;
	cfg.width = scene_params.FindOr<Uint32>("width", 1080);
	cfg.height = scene_params.FindOr<Uint32>("height", 720);
	cfg.max_depth = scene_params.FindOr<Uint32>("max depth", 4);
	cfg.samples_per_pixel = scene_params.FindOr<Uint32>("samples per pixel", 16);

	json camera_json = scene_params.FindJson("camera");
	if (camera_json.is_null())
	{
		AMBER_ERROR("Missing camera parameters in config file!");
		return false;
	}

	JsonParams camera_params(camera_json);

	Float camera_position[3] = { 0.0f, 0.0f, 0.0f };
	camera_params.FindArray("position", camera_position);
	Float look_at[3] = { 0.0f, 0.0f, 1.0f };
	camera_params.FindArray("look_at", look_at);
	cfg.camera.Initialize(Vector3(camera_position), Vector3(look_at));

	Float fovy = camera_params.FindOr<Float>("fov", 45.0f);
	cfg.camera.SetFovY(fovy);
	cfg.camera.SetAspectRatio((Float)cfg.width / cfg.height);
	return true;
}

void ProcessCVarIniFile(Char const* cvar_file)
{
	std::string cvar_ini_path = paths::IniDir + cvar_file;
	std::ifstream cvar_ini_file(cvar_ini_path);
	if (!cvar_ini_file.is_open())
	{
		return;
	}
	std::string line;
	while (std::getline(cvar_ini_file, line))
	{
		if (line.empty() || line[0] == '#') continue;
		g_ConsoleManager.ProcessInput(line);
	}
}
