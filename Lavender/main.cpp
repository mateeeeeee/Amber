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
	std::string scene_file, log_file;
	{
		CLI::App cli_parser{ "Lavender" };
		cli_parser.add_option("--scene", scene_file, "Scene file");
		cli_parser.add_option("--log-file", log_file, "Log file");
		CLI11_PARSE(cli_parser, argc, argv);
		if (log_file.empty()) log_file = "lavender.log";
	}

	g_LogManager.Initialize(log_file.c_str(), lavender::LogLevel::Debug);
	lavender::SDLCheck(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0);
	lavender::CudaCheck(cudaSetDevice(0));
	{
		lavender::Window window(1080, 720, "lavender");
		lavender::Editor editor(window, *g_LogManager.GetEditorSink());
		while (window.Loop())
		{
			editor.Run();
		}
	}
	lavender::CudaCheck(cudaDeviceReset());
	SDL_Quit();
	g_LogManager.Destroy();

	return 0;
}