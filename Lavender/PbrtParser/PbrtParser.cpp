#include "PbrtParser.h"
#include "PbrtLexer.h"
#include "Core/Logger.h"

namespace lavender
{
	using enum PbrtTokenKind;

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
		switch (current_token->GetKind())
		{
		case KW_Camera:
		case KW_LookAt:
		case KW_Integrator:
		case KW_Film:
		case KW_WorldBegin:
		case KW_AttributeBegin:
		case KW_LightSource:
		case KW_Sampler:
		case KW_Material:
		case KW_Shape:
		case KW_Texture:
		case KW_Translate:
		}
	}

	void PbrtParser::ParseCamera()
	{
		Expect(KW_Camera);
		if (current_token->IsNot(identifier))
		{
			//
			return;
		}
		std::string_view data = current_token->GetData();
		if (data == "perspective")
		{

		}
		else
		{
			//not supported
			return;
		}
	}

}
