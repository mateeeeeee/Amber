#define SDL_MAIN_HANDLED
#include "Utilities/SDLUtil.h"
#include "Cuda/CudaUtil.h"
#include "cuda_runtime.h"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"

#include "CLI/CLI.hpp"

#include "Core/Window.h"
#include "Editor/Editor.h"
#include "Editor/EditorSink.h"

int main(int argc, char* argv[])
{
	CLI::App cli_parser{"Lavender"};
	std::string config_file;
	cli_parser.add_option("--config", config_file, "Config file");
	CLI11_PARSE(cli_parser, argc, argv);

	auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
	console_sink->set_level(spdlog::level::trace);
	console_sink->set_pattern("[%^%l%$] %v");
	auto editor_sink = std::make_shared<lavender::EditorSink>();
	editor_sink->set_level(spdlog::level::trace);
	editor_sink->set_pattern("[%^%l%$] %v");
	std::shared_ptr<spdlog::logger> lavender_logger = std::make_shared<spdlog::logger>(std::string("lavender logger"), spdlog::sinks_init_list{ console_sink, editor_sink });
	lavender_logger->set_level(spdlog::level::trace);
	spdlog::set_default_logger(lavender_logger);

	lavender::SDLCheck(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0);
	lavender::CudaCheck(cudaSetDevice(0));
	{
		lavender::Window window(1080, 720, "lavender");
		lavender::Editor editor(window, editor_sink);

		while (window.Loop())
		{
			editor.Run();
		}
	}
	lavender::CudaCheck(cudaDeviceReset());
	SDL_Quit();
	return 0;
}