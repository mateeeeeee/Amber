#pragma once
#include <string>
#include <string_view>
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
		LogCallbackType callbacks[(Uint)LogLevel::Count] = {};

	private:
		LogManager() = default;

		void SetLogCallback(LogLevel level, LogCallbackType&& callback)
		{
			callbacks[(Uint)level] = callback;
		}
	};
	#define g_Log amber::LogManager::Get()

#define AMBER_DEBUG(fmt, ...)  g_Log.CLog(amber::LogLevel::Debug, fmt  __VA_OPT__(,) __VA_ARGS__)
#define AMBER_INFO(fmt, ...)   g_Log.CLog(amber::LogLevel::Info, fmt  __VA_OPT__(,) __VA_ARGS__)
#define AMBER_WARN(fmt, ...)   g_Log.CLog(amber::LogLevel::Warning, fmt  __VA_OPT__(,) __VA_ARGS__)
#define AMBER_ERROR(fmt, ...)  g_Log.CLog(amber::LogLevel::Error, fmt  __VA_OPT__(,) __VA_ARGS__)
}