#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"

int main()
{
	auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
	console_sink->set_level(spdlog::level::trace);
	console_sink->set_pattern("[%^%l%$] %v");

	std::shared_ptr<spdlog::logger> lavender_logger = std::make_shared<spdlog::logger>(std::string("lavender logger"), spdlog::sinks_init_list{ console_sink });
	lavender_logger->set_level(spdlog::level::trace);
	spdlog::set_default_logger(lavender_logger);

	
	return 0;
}