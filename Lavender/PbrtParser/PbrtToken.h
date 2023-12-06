#pragma once
#include <string_view>

namespace lavender
{
	enum class PbrtTokenKind : uint16
	{
	#define TOKEN(X) X,
	#include "PbrtTokens.def"
	};
	std::string_view GetTokenName(PbrtTokenKind t);

	bool IsKeyword(std::string_view identifer);
	PbrtTokenKind GetKeywordType(std::string_view identifer);

	class PbrtToken
	{
	public:
		PbrtToken() : kind(PbrtTokenKind::unknown), data{} {}
		PbrtToken(PbrtTokenKind kind) : kind(kind), data{} {}

		void Reset()
		{
			kind = PbrtTokenKind::unknown;
			data.clear();
		}

		bool Is(PbrtTokenKind t) const { return kind == t; }
		bool IsNot(PbrtTokenKind t) const { return kind != t; }
		template <typename... Ts>
		bool IsOneOf(PbrtTokenKind t1, Ts... ts) const
		{
			if constexpr (sizeof...(Ts) == 0) return Is(t1);
			else return Is(t1) || IsOneOf(ts...);
		}

		PbrtTokenKind GetKind() const { return kind; }
		void SetKind(PbrtTokenKind t) { kind = t; }

		void SetData(char const* p_data, uint64 count)
		{
			data = std::string(p_data, count);
		}
		void SetData(char const* start, char const* end)
		{
			data = std::string(start, end - start);
		}
		void SetData(std::string_view identifier)
		{
			data = std::string(identifier);
		}
		std::string_view GetData() const
		{
			return data;
		}

	private:
		PbrtTokenKind kind;
		std::string data;
	};
}