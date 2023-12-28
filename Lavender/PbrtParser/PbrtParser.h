#pragma once
#include <vector>
#include "PbrtToken.h"

namespace lavender
{
	class PbrtParser
	{
		using TokenPtr = std::vector<PbrtToken>::iterator;
	public:
		PbrtParser() = default;
		~PbrtParser() = default;

		void Parse(char const* scene_file);

	private:
		std::vector<PbrtToken> tokens;
		TokenPtr current_token;
		bool parsing_world_def = false;

	private:

		bool Consume(PbrtTokenKind k);
		template<typename... Ts>
		bool Consume(PbrtTokenKind k, Ts... ts);
		bool Expect(PbrtTokenKind k);
		template<typename... Ts>
		bool Expect(PbrtTokenKind k, Ts... ts);

		void ParseScene();

		void ParseCamera();
	};
}

