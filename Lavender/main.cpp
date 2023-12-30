#define SDL_MAIN_HANDLED
#include "Utilities/SDLUtil.h"
#include "CLI/CLI.hpp"

#include "Core/Window.h"
#include "Core/Logger.h"
#include "Core/Paths.h"
#include "Editor/Editor.h"
#include "Scene/Scene.h"
#include "Scene/Renderer.h"
#include "Utilities/JsonUtil.h"

using namespace lavender;

struct Config
{
	std::string scene_file;
	uint32 width;
	uint32 height;
	uint32 max_depth;
	uint32 samples_per_pixel;
};
bool ParseConfig(char const* config_file, Config& cfg);

int main(int argc, char* argv[])
{
	std::string config_file, log_file;
	bool use_editor = false, maximize_window = false, bool stats_enabled = false;
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
	g_LogManager.Initialize(log_file.c_str(), lavender::LogLevel::Debug);

	Config cfg{};
	bool config_parsed = ParseConfig(config_file.c_str(), cfg);
	if (!config_parsed)
	{
		return EXIT_FAILURE;
	}
	
	std::unique_ptr<Scene> scene = nullptr;
	try
	{
		scene = LoadScene(cfg.scene_file.c_str());
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
	Renderer renderer(std::move(scene));
	if(use_editor)
	{
		Window window(1080, 720, "lavender");
		Editor editor(window, *g_LogManager.GetEditorSink());
		while (window.Loop())
		{
			editor.Run();
		}
	}
	else
	{
		renderer.Render();
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
	cfg.width = scene_params.FindOr<uint32>("width", 1080);
	cfg.height = scene_params.FindOr<uint32>("height", 720);
	cfg.max_depth = scene_params.FindOr<uint32>("max depth", 4);
	cfg.samples_per_pixel = scene_params.FindOr<uint32>("samples per pixel", 16);

	return true;
}
