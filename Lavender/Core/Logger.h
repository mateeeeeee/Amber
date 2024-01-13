#pragma once
#include <string_view>
#include <format>
#include "Utilities/Singleton.h"

namespace lavender
{
	class EditorSink;

	enum class LogLevel : uint8 
	{
		Debug,
		Info,
		Warning,
		Error,
		Count
	};

	class LogManager : public Singleton<LogManager>
	{
		friend class Singleton<LogManager>;
		using LogCallbackType = void (*)(std::string const&);
	public:

		void Initialize(char const* log_file, LogLevel level);
		void Destroy();

		void CLog(LogLevel level, std::string_view fmt, ...);

		template<typename... Args>
		void Log(LogLevel level, std::string_view fmt, Args&&... args)
		{
			std::string msg = std::vformat(fmt, std::make_format_args(args...));
			uint32 i = (uint32)level;
			LAV_ASSERT(callbacks[i]);
			callbacks[i](msg);
		}

		EditorSink* GetEditorSink();

	private:
		LogCallbackType callbacks[(uint32)LogLevel::Count];

	private:
		LogManager() = default;

		void SetLogCallback(LogLevel level, LogCallbackType&& callback)
		{
			callbacks[(uint32)level] = callback;
		}
	};
	#define g_LogManager lavender::LogManager::Get() 

	#define LAV_DEBUG(fmt, ...)  g_LogManager.CLog(lavender::LogLevel::Debug, fmt,  __VA_ARGS__)
	#define LAV_INFO(fmt, ...)   g_LogManager.CLog(lavender::LogLevel::Info, fmt,  __VA_ARGS__)
	#define LAV_WARN(fmt, ...)   g_LogManager.CLog(lavender::LogLevel::Warning, fmt,  __VA_ARGS__)
	#define LAV_ERROR(fmt, ...)  g_LogManager.CLog(lavender::LogLevel::Error, fmt,  __VA_ARGS__)
}