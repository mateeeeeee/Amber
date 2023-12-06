#pragma once
#include <vector>
#include "PbrtToken.h"

namespace lavender
{
	template<typename P>
	concept CharPredicate = requires(P p, char a)
	{
		{ p(a) } -> std::convertible_to<bool>;
	};

	class PbrtLexer
	{
	public:
		PbrtLexer() = default;
		~PbrtLexer() = default;

		void Lex(char const* scene_file);
		std::vector<PbrtToken> GetTokens() const { return tokens; }

	private:
		std::vector<PbrtToken> tokens;
		char const* buf_ptr = nullptr;
		char const* cur_ptr = nullptr;

	private:
		bool LexToken(PbrtToken&);
		bool LexNumber(PbrtToken&);
		bool LexIdentifier(PbrtToken&);
		bool LexString(PbrtToken&);
		bool LexEndOfFile(PbrtToken&);
		bool LexNewLine(PbrtToken&);
		bool LexComment(PbrtToken&);
		bool LexPunctuator(PbrtToken&);

		void UpdatePointers()
		{
			buf_ptr = cur_ptr;
		}

		void FillToken(PbrtToken& t, PbrtTokenKind type, char const* end)
		{
			t.SetKind(type);
			t.SetData(cur_ptr, end);
			cur_ptr = end;
		}
		template<CharPredicate P>
		void Consume(char const*& start, P&& predicate)
		{
			for (; predicate(*start); ++start);
		}
		template<CharPredicate P>
		void FillToken(PbrtToken& t, PbrtTokenKind type, P&& predicate)
		{
			t.SetKind(type);
			char const* tmp_ptr = cur_ptr;
			Consume(tmp_ptr, std::forward<P>(predicate));
			t.SetData(cur_ptr, tmp_ptr);
			cur_ptr = tmp_ptr;
		}
	};
}