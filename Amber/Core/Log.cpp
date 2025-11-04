#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "Log.h"
#include "Core/Paths.h"
#include "Editor/EditorSink.h"


namespace amber
{
	static spdlog::level::level_enum GetSpdlogLevel(LogLevel level)
	{
		switch (level)
		{
		case LogLevel::Debug:	return spdlog::level::debug;
		case LogLevel::Info:	return spdlog::level::info;
		case LogLevel::Warning: return spdlog::level::warn;
		case LogLevel::Error:   return spdlog::level::err;
		case LogLevel::Count:
		default:				
								return spdlog::level::debug;
		}
	}

	void LogManager::Initialize(Char const* log_file, LogLevel log_level)
	{
		spdlog::level::level_enum spdlog_level = GetSpdlogLevel(log_level);
		auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
		console_sink->set_level(spdlog_level);
		console_sink->set_pattern("[%^%l%$] %v");
		auto editor_sink = std::make_shared<EditorSink>();
		editor_sink->set_level(spdlog_level);
		editor_sink->set_pattern("[%^%l%$] %v");
		auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(paths::LogDir + log_file);
		file_sink->set_level(spdlog_level);
		file_sink->set_pattern("[%^%l%$] %v");
		std::shared_ptr<spdlog::logger> amber_logger = std::make_shared<spdlog::logger>(std::string("amber logger"), spdlog::sinks_init_list{ console_sink, editor_sink, file_sink });
		amber_logger->set_level(spdlog_level);
		spdlog::set_default_logger(amber_logger);

		SetLogCallback(LogLevel::Debug, spdlog::debug<std::string>);
		SetLogCallback(LogLevel::Info, spdlog::info<std::string>);
		SetLogCallback(LogLevel::Warning, spdlog::warn<std::string>);
		SetLogCallback(LogLevel::Error, spdlog::error<std::string>);
	}

	void LogManager::Destroy()
	{
		spdlog::set_default_logger(nullptr);
	}

	void LogManager::CLog(LogLevel level, std::string_view fmt, ...)
	{
		va_list args;
		va_start(args, fmt);
		size_t size = std::vsnprintf(nullptr, 0, fmt.data(), args) + 1;
		std::string msg(size, '\0');
		std::vsnprintf(&msg[0], size, fmt.data(), args);
		va_end(args);
		Uint32 i = static_cast<Uint32>(level);
		AMBER_ASSERT(callbacks[i]);
		callbacks[i](msg);
	}

	EditorSink* LogManager::GetEditorSink()
	{
		for (auto& sink : spdlog::default_logger()->sinks())
		{
			if (auto editor_sink = dynamic_pointer_cast<EditorSink>(sink)) return editor_sink.get();
		}
		AMBER_ASSERT(false);
		return nullptr;
	}

}