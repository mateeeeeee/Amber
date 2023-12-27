#include "PbrtParser.h"
#include "PbrtLexer.h"
#include "Core/Logger.h"

namespace lavender
{
	void PbrtParser::Parse(char const* scene_file)
	{
		PbrtLexer pbrt_lexer;
		pbrt_lexer.Lex(scene_file);
		tokens = pbrt_lexer.GetTokens(); 
	}

	bool PbrtParser::Consume(PbrtTokenKind k)
	{
		if (current_token->Is(k))
		{
			++current_token; return true;
		}
		else return false;
	}
	template<typename... Ts>
	bool PbrtParser::Consume(PbrtTokenKind k, Ts... ts)
	{
		if (current_token->IsOneOf(k, ts...))
		{
			++current_token; return true;
		}
		else return false;
	}

	bool PbrtParser::Expect(PbrtTokenKind k)
	{
		if (!Consume(k))
		{
			LAVENDER_ERROR("Parser error!");
			return false;
		}
		return true;
	}
	template<typename... Ts>
	bool PbrtParser::Expect(PbrtTokenKind k, Ts... ts)
	{
		if (!Consume(k, ts...))
		{
			LAVENDER_ERROR("Parser error!");
			return false;
		}
		return true;
	}

	void PbrtParser::ParseScene()
	{

	}

}
