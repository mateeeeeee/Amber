#pragma once
#include <string_view>
#include <format>
#include "Utilities/Singleton.h"

namespace amber
{
	class EditorSink;

	enum class LogLevel : Uint8
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

		void Initialize(Char const* log_file, LogLevel level);
		void Destroy();
		void CLog(LogLevel level, std::string_view fmt, ...);

		EditorSink* GetEditorSink();

	private:
		LogCallbackType callbacks[(Uint32)LogLevel::Count] = {};

	private:
		LogManager() = default;

		void SetLogCallback(LogLevel level, LogCallbackType&& callback)
		{
			callbacks[(Uint32)level] = callback;
		}
	};
#define g_LogManager amber::LogManager::Get() 

#define AMBER_DEBUG(fmt, ...)  g_LogManager.CLog(amber::LogLevel::Debug, fmt,  __VA_ARGS__)
#define AMBER_INFO(fmt, ...)   g_LogManager.CLog(amber::LogLevel::Info, fmt,  __VA_ARGS__)
#define AMBER_WARN(fmt, ...)   g_LogManager.CLog(amber::LogLevel::Warning, fmt,  __VA_ARGS__)
#define AMBER_ERROR(fmt, ...)  g_LogManager.CLog(amber::LogLevel::Error, fmt,  __VA_ARGS__)
}