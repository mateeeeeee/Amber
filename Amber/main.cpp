#define SDL_MAIN_HANDLED
#include <fstream>
#include "Utilities/SDLUtil.h"
#include "Core/Window.h"
#include "Core/Log.h"
#include "Core/Paths.h"
#include "Core/CommandLineOptions.h"
#include "Core/ConsoleManager.h"
#include "Editor/Editor.h"
#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include "Optix/OptixPathTracer.h"
#include "Utilities/CpuBuffer2D.h"
#include "Utilities/JsonUtil.h"

using namespace amber;

struct SceneConfig
{
	Camera camera;
	std::string scene_environment;
	std::string model_file;
	Float		model_scale;
	Uint32		width;
	Uint32		height;
	PathTracerConfig path_tracer_config;
};
Bool ParseSceneConfig(Char const* config_file, SceneConfig& cfg);
void ProcessCVarIniFile(Char const* cvar_file);

int main(Int argc, Char* argv[])
{
	CommandLineOptions::Initialize(argc, argv);
#ifdef _DEBUG
	g_LogManager.Initialize(log_file.c_str(), LogLevel::Debug);
#else 
	g_LogManager.Initialize(CommandLineOptions::GetLogFile().c_str(), LogLevel::Error);
#endif

	SceneConfig cfg{};
	if (!ParseSceneConfig(CommandLineOptions::GetConfigFile().c_str(), cfg))
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
	Uint32 windowWidth = cfg.width;
	Uint32 windowHeight = cfg.height;
	OptixPathTracer path_tracer(windowWidth, windowHeight, cfg.path_tracer_config, std::move(scene));
	ProcessCVarIniFile("cvars.ini");
	if(CommandLineOptions::GetUseEditor())
	{
		Window window(windowWidth, windowHeight, "amber");
		if (CommandLineOptions::GetMaximizeWindow())
		{
			window.Maximize();
		}
		Editor editor(window, camera, path_tracer);
		editor.SetEditorSink(g_LogManager.GetEditorSink());
		while (window.Loop())
		{
			editor.Run();
		}
	}
	else
	{
		path_tracer.Render(camera);
		path_tracer.WriteFramebuffer(CommandLineOptions::GetOutputFile().c_str());
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
	cfg.path_tracer_config.max_depth = scene_params.FindOr<Uint32>("max depth", 4);
	cfg.path_tracer_config.samples_per_pixel = scene_params.FindOr<Uint32>("samples per pixel", 16);
	cfg.path_tracer_config.use_denoiser = scene_params.FindOr<Bool>("denoise", false);
	cfg.path_tracer_config.accumulate = scene_params.FindOr<Bool>("accumulate", true);

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
