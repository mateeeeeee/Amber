#include <fstream>
#include <sstream>
#include "PbrtLexer.h"
#include "Core/Logger.h"


namespace lavender
{
	using enum PbrtTokenKind;

	void PbrtLexer::Lex(char const* scene_file)
	{
		std::string scene_path(scene_file);
		std::ifstream input_stream(scene_path);
		auto good = input_stream.good();
		std::ostringstream buf;
		buf << input_stream.rdbuf();
		std::string scene_data = buf.str();
		scene_data.push_back('\0');

		buf_ptr = scene_data.c_str();
		cur_ptr = buf_ptr;

		PbrtToken current_token{};
		do
		{
			current_token.Reset();
			bool result = LexToken(current_token);
			if (!result) return;
			tokens.push_back(current_token);
		} while (current_token.IsNot(eof));
	}

	bool PbrtLexer::LexToken(PbrtToken& token)
	{
		UpdatePointers();
		if ((*cur_ptr == ' ') || (*cur_ptr == '\t'))
		{
			++cur_ptr;
			while ((*cur_ptr == ' ') || (*cur_ptr == '\t')) ++cur_ptr;
			UpdatePointers();
		}

		char c = *cur_ptr++;
		switch (c)
		{
		case '\0':
			return LexEndOfFile(token);
		case '\n':
		{
			bool ret = LexNewLine(token);
			buf_ptr = cur_ptr;
			return ret;
		}
		case '#':
		{
			return LexComment(token);
		}
		case '"':
		{
			return LexString(token);
		}
		case '.':
		{
			--cur_ptr;
			if (std::isdigit(*(cur_ptr + 1)))
			{
				return LexNumber(token);
			}
			else return LexPunctuator(token);
		}
		case '0': case '1': case '2': case '3': case '4':
		case '5': case '6': case '7': case '8': case '9':
		{
			--cur_ptr;
			return LexNumber(token);
		}
		case 'A': case 'B': case 'C': case 'D': case 'E': case 'F': case 'G':
		case 'H': case 'I': case 'J': case 'K': case 'L': case 'M': case 'N':
		case 'O': case 'P': case 'Q': case 'R': case 'S': case 'T': case 'U':
		case 'V': case 'W': case 'X': case 'Y': case 'Z':
		case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g':
		case 'h': case 'i': case 'j': case 'k': case 'l': case 'm': case 'n':
		case 'o': case 'p': case 'q': case 'r': case 's': case 't': case 'u':
		case 'v': case 'w': case 'x': case 'y': case 'z':
		case '_':
		{
			--cur_ptr;
			return LexIdentifier(token);
		}
		case '[': case ']': 
		{
			--cur_ptr;
			return LexPunctuator(token);
		}
		}
		LAVENDER_ERROR("Lexer error!");
		return false;
	}

	bool PbrtLexer::LexNumber(PbrtToken& t)
	{
		char const* tmp_ptr = cur_ptr;
		Consume(tmp_ptr, [](char c) -> bool { return std::isdigit(c); });
		if (*tmp_ptr == '.')
		{
			tmp_ptr++;
			Consume(tmp_ptr, [](char c) -> bool { return std::isdigit(c); });
			if (std::isalpha(*tmp_ptr)) return false;
			FillToken(t, number, tmp_ptr);
			UpdatePointers();
			return true;
		}
		else if (std::isalpha(*tmp_ptr))
		{
			UpdatePointers();
			return false;
		}
		else
		{
			FillToken(t, number, tmp_ptr);
			UpdatePointers();
			return true;
		}
		return true;
	}

	bool PbrtLexer::LexIdentifier(PbrtToken& t)
	{
		FillToken(t, identifier, [](char c) -> bool { return std::isalnum(c) || c == '_'; });
		std::string_view identifier = t.GetData();
		if (IsKeyword(identifier))
		{
			t.SetKind(GetKeywordType(identifier));
		}
		UpdatePointers();
		return true;
	}

	bool PbrtLexer::LexString(PbrtToken& t)
	{
		FillToken(t, string, [](char c) -> bool { return c != '"'; });
		++cur_ptr;
		UpdatePointers();
		return true;
	}

	bool PbrtLexer::LexEndOfFile(PbrtToken& t)
	{
		t.SetKind(eof);
		return true;
	}

	bool PbrtLexer::LexNewLine(PbrtToken& t)
	{
		t.SetKind(newline);
		return true;
	}

	bool PbrtLexer::LexComment(PbrtToken& t)
	{
		FillToken(t, comment, [](char c) -> bool { return c != '\n' && c != '\0'; });
		UpdatePointers();
		return true;
	}

	bool PbrtLexer::LexPunctuator(PbrtToken& t)
	{
		char c = *cur_ptr++;
		switch (c)
		{
		case '[':
			t.SetKind(left_square);
			break;
		case ']':
			t.SetKind(right_square);
			break;
		}
		UpdatePointers();
		return true;
	}

}

