#include "PbrtToken.h"


namespace lavender
{



	std::string_view GetTokenName(PbrtTokenKind t)
	{
		return "";
	}

	bool IsKeyword(std::string_view identifer)
	{
		return false;
	}

	PbrtTokenKind GetKeywordType(std::string_view identifer)
	{
		return PbrtTokenKind::unknown;
	}

}
