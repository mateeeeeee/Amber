#pragma once
#include <memory>
#include "spdlog/spdlog.h"
#include "Core/Log.h"

namespace amber
{
	class EditorSink : public spdlog::sinks::sink 
	{
	public:
		EditorSink();
		~EditorSink();

		virtual void log(const spdlog::details::log_msg& msg) override;
		virtual void flush() override {}
		virtual void set_pattern(const std::string& pattern) override {}
		virtual void set_formatter(std::unique_ptr<spdlog::formatter> sink_formatter)  override {}

		void Draw(const Char* title, Bool* p_open = nullptr);

	private:
		std::unique_ptr<struct ImGuiLogger> imgui_log;
	};
}