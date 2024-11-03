#include <algorithm>
#include <sstream>
#include "StringUtil.h"

namespace amber
{
	std::wstring ToWideString(std::string const& in)
	{
		std::wstring out{};
		out.reserve(in.length());
		const Char* ptr = in.data();
		const Char* const end = in.data() + in.length();

		mbstate_t state{};
		wchar_t wc;
		while (size_t len = mbrtowc(&wc, ptr, end - ptr, &state))
		{
			if (len == static_cast<size_t>(-1)) // bad encoding
				return std::wstring{};
			if (len == static_cast<size_t>(-2))
				break;
			out.push_back(wc);
			ptr += len;
		}
		return out;
	}
	std::string ToString(std::wstring const& in)
	{
		std::string out{};
		out.reserve(MB_CUR_MAX * in.length());

		mbstate_t state{};
		for (wchar_t wc : in)
		{
			Char mb[8]{}; // ensure null-terminated for UTF-8 (maximum 4 byte)
			const auto len = wcrtomb(mb, wc, &state);
			out += std::string_view{ mb, len };
		}
		return out;
	}

	std::string ToLower(std::string const& in)
	{
		std::string out; out.resize(in.size());
		std::transform(std::begin(in), std::end(in), std::begin(out), [](Char c) {return std::tolower(c); });
		return out;
	}
	std::string ToUpper(std::string const& in)
	{
		std::string out; out.resize(in.size());
		std::transform(std::begin(in), std::end(in), std::begin(out), [](Char c) {return std::toupper(c); });
		return out;
	}


	Bool FromCString(const Char* in, int& out)
	{
		std::istringstream iss(in);
		iss >> out;
		return !iss.fail() && iss.eof();
	}

	Bool FromCString(const Char* in, Float& out)
	{
		std::istringstream iss(in);
		iss >> out;
		return !iss.fail() && iss.eof();
	}

	Bool FromCString(const Char* in, std::string& out)
	{
		out = in;
		return true;
	}

	Bool FromCString(const Char* in, Bool& out)
	{
		std::string str(in);
		std::transform(str.begin(), str.end(), str.begin(), ::tolower);
		if (str == "0" || str == "false")
		{
			out = false;
			return true;
		}
		else if (str == "1" || str == "true")
		{
			out = true;
			return true;
		}
		return false;
	}

	Bool FromCString(Char const* in, Vector3& out)
	{
		std::istringstream iss(in);
		Char open_parenthesis, comma1, comma2, close_parenthesis;
		if (iss >> open_parenthesis >> out.x >> comma1 >> out.y >> comma2 >> out.z >> close_parenthesis)
		{
			return open_parenthesis == '(' && comma1 == ',' && comma2 == ',' && close_parenthesis == ')' && iss.eof();
		}
		return false;
	}

	std::string IntToString(int val)
	{
		return std::to_string(val);
	}
	std::string FloatToString(Float val)
	{
		return std::to_string(val);
	}
	std::string CStrToString(Char const* val)
	{
		return val;
	}
	std::string BoolToString(Bool val)
	{
		return val ? "true" : "false";
	}

	std::string Vector3ToString(Vector3 const& val)
	{
		return "(" + std::to_string(val.x) + "," + std::to_string(val.y) + "," + std::to_string(val.z) + ")";
	}

	std::vector<std::string> SplitString(const std::string& text, Char delimeter)
	{
		std::vector<std::string> tokens;
		size_t start = 0, end = 0;
		while ((end = text.find(delimeter, start)) != std::string::npos)
		{
			tokens.push_back(text.substr(start, end - start));
			start = end + 1;
		}
		tokens.push_back(text.substr(start));
		return tokens;
	}
}