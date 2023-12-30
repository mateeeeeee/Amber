#pragma once
#include "Utilities/Singleton.h"
#include "spdlog/spdlog.h"

namespace lavender
{
	class EditorSink;

	enum class LogLevel : uint8 
	{
		Debug,
		Info,
		Warning,
		Error
	};

	class LogManager : public Singleton<LogManager>
	{
		friend class Singleton<LogManager>;
	public:

		void Initialize(char const* log_file, LogLevel level);
		void Destroy();

		template<typename... Args>
		void Log(LogLevel level, std::string_view fmt, Args&&... args)
		{
			std::string msg = std::vformat(fmt, std::make_format_args(args...));
			switch (level)
			{
			case LogLevel::Debug:	return spdlog::debug(msg);
			case LogLevel::Info:	return spdlog::info(msg);
			case LogLevel::Warning:	return spdlog::warn(msg);
			case LogLevel::Error:	return spdlog::error(msg);
			}
		}
		EditorSink* GetEditorSink();

	private:
		LogManager() = default;
	};
	#define g_LogManager lavender::LogManager::Get() 

	#define LAV_DEBUG(fmt, ...)  g_LogManager.Log(lavender::LogLevel::Debug, fmt,  __VA_ARGS__)
	#define LAV_INFO(fmt, ...)   g_LogManager.Log(lavender::LogLevel::Info, fmt,  __VA_ARGS__)
	#define LAV_WARN(fmt, ...)   g_LogManager.Log(lavender::LogLevel::Warning, fmt,  __VA_ARGS__)
	#define LAV_ERROR(fmt, ...)  g_LogManager.Log(lavender::LogLevel::Error, fmt,  __VA_ARGS__)
}