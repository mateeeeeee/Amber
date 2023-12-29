#define SDL_MAIN_HANDLED
#include "Utilities/SDLUtil.h"
#include "Cuda/CudaUtil.h"
#include "cuda_runtime.h"
#include "CLI/CLI.hpp"

#include "Core/Window.h"
#include "Core/Logger.h"
#include "Editor/Editor.h"


int main(int argc, char* argv[])
{
	std::string config_file, log_file;
	bool use_editor = false;
	{
		CLI::App cli_parser{ "Lavender" };
		cli_parser.add_option("--config-file", config_file, "Config file");
		cli_parser.add_option("--log-file", log_file, "Log file");
		CLI::Option* use_editor_opt = cli_parser.add_flag("--editor", "Use editor");
		CLI11_PARSE(cli_parser, argc, argv);
		if (log_file.empty()) log_file = "lavender.log";
		use_editor = (bool)*use_editor_opt;
	}

	g_LogManager.Initialize(log_file.c_str(), lavender::LogLevel::Debug);
	lavender::CudaCheck(cudaSetDevice(0));
	//Renderer renderer(config_file);
	if(use_editor)
	{
		lavender::SDLCheck(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0);
		lavender::Window window(1080, 720, "lavender");
		lavender::Editor editor(window, *g_LogManager.GetEditorSink());
		while (window.Loop())
		{
			editor.Run();
		}
		SDL_Quit();
	}
	else
	{

	}
	lavender::CudaCheck(cudaDeviceReset());
	g_LogManager.Destroy();

	return 0;
}