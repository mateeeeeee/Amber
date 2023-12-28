#include "Logger.h"
#include "Core/Paths.h"
#include "Editor/EditorSink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"

namespace lavender
{
	static spdlog::level::level_enum GetSpdlogLevel(LogLevel level)
	{
		switch (level)
		{
		case LogLevel::Debug: return spdlog::level::debug;
		case LogLevel::Info:  return spdlog::level::info;
		case LogLevel::Warning: return spdlog::level::warn;
		case LogLevel::Error:   return spdlog::level::err;
		}
		return spdlog::level::debug;
	}

	void LogManager::Initialize(char const* log_file, LogLevel log_level)
	{
		spdlog::level::level_enum spdlog_level = GetSpdlogLevel(log_level);
		auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
		console_sink->set_level(spdlog_level);
		console_sink->set_pattern("[%^%l%$] %v");
		auto editor_sink = std::make_shared<EditorSink>();
		editor_sink->set_level(spdlog_level);
		editor_sink->set_pattern("[%^%l%$] %v");
		auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(paths::LogDir() + log_file);
		file_sink->set_level(spdlog_level);
		file_sink->set_pattern("[%^%l%$] %v");
		std::shared_ptr<spdlog::logger> lavender_logger = std::make_shared<spdlog::logger>(std::string("lavender logger"), spdlog::sinks_init_list{ console_sink, editor_sink, file_sink });
		lavender_logger->set_level(spdlog_level);
		spdlog::set_default_logger(lavender_logger);
	}

	void LogManager::Destroy()
	{
		spdlog::set_default_logger(nullptr);
	}

	EditorSink* LogManager::GetEditorSink()
	{
		for (auto& sink : spdlog::default_logger()->sinks())
		{
			if (auto editor_sink = dynamic_pointer_cast<EditorSink>(sink)) return editor_sink.get();
		}
		LAVENDER_ASSERT(false);
		return nullptr;
	}

}

