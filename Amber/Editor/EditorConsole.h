#pragma once
#include "ImGui/imgui.h"

namespace amber
{
	class EditorConsole
	{
	public:
		EditorConsole();
		~EditorConsole();

		void Draw(const Char* title, Bool* p_open = nullptr);
		void DrawBasic(const Char* title, Bool* p_open = nullptr);

	private:
		Char                  InputBuf[256];
		ImVector<Char*>       Items;
		ImVector<const Char*> Commands;
		ImVector<const Char*> CommandDescriptions;
		ImVector<Char*>       History;
		int                   HistoryPos;
		ImGuiTextFilter       Filter;
		Bool                  AutoScroll;
		Bool                  ScrollToBottom;
		int					  CursorPos;

	private:
		void    ClearLog();
		void    AddLog(const Char* fmt, ...) IM_FMTARGS(2);

		void ExecCommand(const Char* cmd);
		int	 TextEditCallback(ImGuiInputTextCallbackData* data);
	};
}

