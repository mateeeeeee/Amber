#include "ImGui/imgui.h"
#include "EditorSink.h"


namespace lavender
{
	struct ImGuiLogger
	{
		ImGuiTextBuffer     Buf;
		ImGuiTextFilter     Filter;
		ImVector<int>       LineOffsets;
		bool                AutoScroll;

		ImGuiLogger()
		{
			AutoScroll = true;
			Clear();
		}

		void Clear()
		{
			Buf.clear();
			LineOffsets.clear();
			LineOffsets.push_back(0);
		}
		void AddLog(const char* fmt, ...) IM_FMTARGS(2)
		{
			int old_size = Buf.size();
			va_list args;
			va_start(args, fmt);
			Buf.appendfv(fmt, args);
			va_end(args);
			for (int new_size = Buf.size(); old_size < new_size; old_size++)
				if (Buf[old_size] == '\n')
					LineOffsets.push_back(old_size + 1);
		}
		void Draw(const char* title, bool* p_open = NULL)
		{
			if (!ImGui::Begin(title, p_open))
			{
				ImGui::End();
				return;
			}

			// Options menu
			if (ImGui::BeginPopup("Options"))
			{
				ImGui::Checkbox("Auto-scroll", &AutoScroll);
				ImGui::EndPopup();
			}

			// Main window
			if (ImGui::Button("Options"))
				ImGui::OpenPopup("Options");
			ImGui::SameLine();
			bool clear = ImGui::Button("Clear");
			ImGui::SameLine();
			bool copy = ImGui::Button("Copy");
			ImGui::SameLine();
			Filter.Draw("Filter", -100.0f);

			ImGui::Separator();
			ImGui::BeginChild("scrolling", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

			if (clear)
				Clear();
			if (copy)
				ImGui::LogToClipboard();

			ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
			const char* buf = Buf.begin();
			const char* buf_end = Buf.end();
			if (Filter.IsActive())
			{
				for (int line_no = 0; line_no < LineOffsets.Size; line_no++)
				{
					const char* line_start = buf + LineOffsets[line_no];
					const char* line_end = (line_no + 1 < LineOffsets.Size) ? (buf + LineOffsets[line_no + 1] - 1) : buf_end;
					if (Filter.PassFilter(line_start, line_end))
						ImGui::TextUnformatted(line_start, line_end);
				}
			}
			else
			{
				ImGuiListClipper clipper;
				clipper.Begin(LineOffsets.Size);
				while (clipper.Step())
				{
					for (int line_no = clipper.DisplayStart; line_no < clipper.DisplayEnd; line_no++)
					{
						const char* line_start = buf + LineOffsets[line_no];
						const char* line_end = (line_no + 1 < LineOffsets.Size) ? (buf + LineOffsets[line_no + 1] - 1) : buf_end;
						ImGui::TextUnformatted(line_start, line_end);
					}
				}
				clipper.End();
			}
			ImGui::PopStyleVar();

			if (AutoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
				ImGui::SetScrollHereY(1.0f);

			ImGui::EndChild();
			ImGui::End();
		}
	};

	EditorSink::EditorSink() : imgui_log(new ImGuiLogger{}) {}

	EditorSink::~EditorSink() = default;

	void EditorSink::log(spdlog::details::log_msg const& msg)
	{
		spdlog::log_clock::time_point tp = msg.time;
		std::time_t tm_time = spdlog::log_clock::to_time_t(tp);
		std::string time_str = std::string(ctime(&tm_time));

		spdlog::level::level_enum level = msg.level;
		std::string level_string = spdlog::level::to_string_view(level).data();

		spdlog::string_view_t message = msg.payload;
		std::string log_message = "[" + level_string + "]" + "[" + time_str + "]" + std::string(message.data());
		imgui_log->AddLog(log_message.c_str());
	}
	void EditorSink::Draw(const char* title, bool* p_open)
	{
		imgui_log->Draw(title, p_open);
	}
}


