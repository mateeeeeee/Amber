#include "EditorConsole.h"
#include "Core/ConsoleManager.h"
#include "Core/ConsoleVariable.h" 
#include "Core/ConsoleCommand.h"

namespace lavender
{
	static int Stricmp(const char* s1, const char* s2)
	{
		int d; while ((d = toupper(*s2) - toupper(*s1)) == 0 && *s1) { s1++; s2++; } return d;
	}
	static int Strnicmp(const char* s1, const char* s2, int n)
	{
		int d = 0; while (n > 0 && (d = toupper(*s2) - toupper(*s1)) == 0 && *s1) { s1++; s2++; n--; } return d;
	}
	static char* Strdup(const char* s)
	{
		IM_ASSERT(s); uint64 len = strlen(s) + 1; void* buf = malloc(len); IM_ASSERT(buf); return (char*)memcpy(buf, (const void*)s, len);
	}
	static void Strtrim(char* s)
	{
		char* str_end = s + strlen(s); while (str_end > s && str_end[-1] == ' ') str_end--; *str_end = 0;
	}

	EditorConsole::EditorConsole()
	{
		ClearLog();
		memset(InputBuf, 0, sizeof(InputBuf));
		HistoryPos = -1;

		Commands.push_back("help");
		Commands.push_back("history");
		Commands.push_back("clear");

		auto RegisterVariables = [&](IConsoleVariable* cvar)
			{
				Commands.push_back(cvar->GetName());
			};
		ConsoleManager::ForEachCVar(RegisterVariables);
		auto RegisterCommands = [&](IConsoleCommand* ccmd)
			{
				Commands.push_back(ccmd->GetName());
			};
		ConsoleManager::ForEachCCmd(RegisterCommands);

		AutoScroll = true;
		ScrollToBottom = false;
	}
	EditorConsole::~EditorConsole()
	{
		ClearLog();
		for (int i = 0; i < History.Size; i++)
			free(History[i]);
	}

	void EditorConsole::Draw(const char* title, bool* p_open)
	{
		ImGui::SetNextWindowSize(ImVec2(520, 600), ImGuiCond_FirstUseEver);
		if (!ImGui::Begin(title, p_open))
		{
			ImGui::End();
			return;
		}
		ImGui::SameLine();
		if (ImGui::SmallButton("Clear")) { ClearLog(); }
		ImGui::SameLine();
		bool copy_to_clipboard = ImGui::SmallButton("Copy");

		ImGui::Separator();

		if (ImGui::BeginPopup("Options"))
		{
			ImGui::Checkbox("Auto-scroll", &AutoScroll);
			ImGui::EndPopup();
		}

		if (ImGui::Button("Options")) ImGui::OpenPopup("Options");
		ImGui::SameLine();
		Filter.Draw("Filter (\"incl,-excl\") (\"error\")", 180);
		ImGui::Separator();

		// Reserve enough left-over height for 1 separator + 1 input text
		const float footer_height_to_reserve = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
		ImGui::BeginChild("ScrollingRegion", ImVec2(0, -footer_height_to_reserve), false, ImGuiWindowFlags_HorizontalScrollbar);
		if (ImGui::BeginPopupContextWindow())
		{
			if (ImGui::Selectable("Clear")) ClearLog();
			ImGui::EndPopup();
		}

		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 1)); // Tighten spacing
		if (copy_to_clipboard) ImGui::LogToClipboard();
		for (int i = 0; i < Items.Size; i++)
		{
			const char* item = Items[i];
			if (!Filter.PassFilter(item)) continue;
			ImGui::TextUnformatted(item);
		}
		if (copy_to_clipboard) ImGui::LogFinish();

		if (ScrollToBottom || (AutoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY()))
			ImGui::SetScrollHereY(1.0f);
		ScrollToBottom = false;

		ImGui::PopStyleVar();
		ImGui::EndChild();
		ImGui::Separator();

		auto TextEditCallbackStub = [](ImGuiInputTextCallbackData* data)
			{
				EditorConsole* console = (EditorConsole*)data->UserData;
				return console->TextEditCallback(data);
			};
		// Command-line
		bool reclaim_focus = false;
		ImGuiInputTextFlags input_text_flags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackCompletion | ImGuiInputTextFlags_CallbackHistory;
		if (ImGui::InputText("Input", InputBuf, IM_ARRAYSIZE(InputBuf), input_text_flags, TextEditCallbackStub, (void*)this))
		{
			char* s = InputBuf;
			Strtrim(s);
			if (s[0]) ExecCommand(s);
			strcpy(s, "");
			reclaim_focus = true;
		}

		// Auto-focus on window apparition
		ImGui::SetItemDefaultFocus();
		if (reclaim_focus)
			ImGui::SetKeyboardFocusHere(-1); // Auto focus previous widget

		ImGui::End();
	}
	void EditorConsole::ClearLog()
	{
		for (int i = 0; i < Items.Size; i++)
			free(Items[i]);
		Items.clear();
	}
	void EditorConsole::AddLog(const char* fmt, ...) IM_FMTARGS(2)
	{
		// FIXME-OPT
		char buf[1024];
		va_list args;
		va_start(args, fmt);
		vsnprintf(buf, IM_ARRAYSIZE(buf), fmt, args);
		buf[IM_ARRAYSIZE(buf) - 1] = 0;
		va_end(args);
		Items.push_back(Strdup(buf));
	}
	void EditorConsole::ExecCommand(const char* cmd)
	{
		AddLog("# %s\n", cmd);
		HistoryPos = -1;
		for (int i = History.Size - 1; i >= 0; i--)
		{
			if (Stricmp(History[i], cmd) == 0)
			{
				free(History[i]);
				History.erase(History.begin() + i);
				break;
			}
		}
		History.push_back(Strdup(cmd));

		if (Stricmp(cmd, "clear") == 0)
		{
			ClearLog();
		}
		else if (Stricmp(cmd, "help") == 0)
		{
			AddLog("Commands:");
			for (int i = 0; i < Commands.Size; i++) AddLog("- %s", Commands[i]); //add cvar descriptions
		}
		else if (Stricmp(cmd, "history") == 0)
		{
			int first = History.Size - 10;
			for (int i = first > 0 ? first : 0; i < History.Size; i++)
				AddLog("%3d: %s\n", i, History[i]);
		}
		else if (!ConsoleManager::Execute(cmd))
		{
			AddLog("Unknown command: '%s'\n", cmd);
		}

		ScrollToBottom = true;
	}
	int EditorConsole::TextEditCallback(ImGuiInputTextCallbackData* data)
	{
		switch (data->EventFlag)
		{
		case ImGuiInputTextFlags_CallbackCompletion:
		{
			const char* word_end = data->Buf + data->CursorPos;
			const char* word_start = word_end;
			while (word_start > data->Buf)
			{
				const char c = word_start[-1];
				if (c == ' ' || c == '\t' || c == ',' || c == ';')
					break;
				word_start--;
			}

			ImVector<const char*> candidates;
			for (int i = 0; i < Commands.Size; i++)
				if (Strnicmp(Commands[i], word_start, (int)(word_end - word_start)) == 0)
					candidates.push_back(Commands[i]);

			if (candidates.Size == 0)
			{
				// No match
				AddLog("No match for \"%.*s\"!\n", (int)(word_end - word_start), word_start);
			}
			else if (candidates.Size == 1)
			{
				// Single match. Delete the beginning of the word and replace it entirely so we've got nice casing.
				data->DeleteChars((int)(word_start - data->Buf), (int)(word_end - word_start));
				data->InsertChars(data->CursorPos, candidates[0]);
				data->InsertChars(data->CursorPos, " ");
			}
			else if (candidates.Size != Commands.Size)
			{
				int match_len = (int)(word_end - word_start);
				for (;;)
				{
					int c = 0;
					bool all_candidates_matches = true;
					for (int i = 0; i < candidates.Size && all_candidates_matches; i++)
						if (i == 0)
							c = toupper(candidates[i][match_len]);
						else if (c == 0 || c != toupper(candidates[i][match_len]))
							all_candidates_matches = false;
					if (!all_candidates_matches)
						break;
					match_len++;
				}

				if (match_len > 0)
				{
					data->DeleteChars((int)(word_start - data->Buf), (int)(word_end - word_start));
					data->InsertChars(data->CursorPos, candidates[0], candidates[0] + match_len);
				}

				// List matches
				AddLog("Possible matches:\n");
				for (int i = 0; i < candidates.Size; i++)
					AddLog("- %s\n", candidates[i]);
			}

			break;
		}
		case ImGuiInputTextFlags_CallbackHistory:
		{
			// Example of HISTORY
			const int prev_history_pos = HistoryPos;
			if (data->EventKey == ImGuiKey_UpArrow)
			{
				if (HistoryPos == -1)
					HistoryPos = History.Size - 1;
				else if (HistoryPos > 0)
					HistoryPos--;
			}
			else if (data->EventKey == ImGuiKey_DownArrow)
			{
				if (HistoryPos != -1)
					if (++HistoryPos >= History.Size)
						HistoryPos = -1;
			}

			// A better implementation would preserve the data on the current input line along with cursor position.
			if (prev_history_pos != HistoryPos)
			{
				const char* history_str = (HistoryPos >= 0) ? History[HistoryPos] : "";
				data->DeleteChars(0, data->BufTextLen);
				data->InsertChars(0, history_str);
			}
		}
		}
		return 0;
	}
}
