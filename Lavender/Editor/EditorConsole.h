#pragma once
#include "imgui.h"

namespace lavender
{
	class EditorConsole
	{
	public:
		EditorConsole();
		~EditorConsole();

		void Draw(const char* title, bool* p_open = nullptr);

	private:
		char                  InputBuf[256];
		ImVector<char*>       Items;
		ImVector<const char*> Commands;
		ImVector<char*>       History;
		int                   HistoryPos;
		ImGuiTextFilter       Filter;
		bool                  AutoScroll;
		bool                  ScrollToBottom;

	private:
		void    ClearLog();
		void    AddLog(const char* fmt, ...) IM_FMTARGS(2);

		void ExecCommand(const char* cmd);
		int	TextEditCallback(ImGuiInputTextCallbackData* data);
	};
}

