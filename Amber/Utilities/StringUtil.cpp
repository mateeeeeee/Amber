#include <algorithm>
#include "StringUtil.h"

namespace amber
{
	std::string ToLower(std::string const& in)
	{
		std::string out; out.resize(in.size());
		std::transform(std::begin(in), std::end(in), std::begin(out), [](char c) {return std::tolower(c); });
		return out;
	}
	std::string ToUpper(std::string const& in)
	{
		std::string out; out.resize(in.size());
		std::transform(std::begin(in), std::end(in), std::begin(out), [](char c) {return std::toupper(c); });
		return out;
	}

	bool FromCString(const char* in, int& out)
	{
		size_t idx = 0;
		char sign = 1;
		out = 0;
		while (*in != '\0')
		{
			if (idx == 0 && *in == '-')
			{
				sign = -1;
			}
			else if (*in && *in >= '0' && *in <= '9')
			{
				out *= 10;
				out += *in - '0';
			}
			else
			{
				return false;
			}

			++in;
			++idx;
		}
		out *= sign;
		return true;
	}
	bool FromCString(const char* in, float& out)
	{
		size_t idx = 0;
		char sign = 1;
		char comma = 0;
		int divisor = 1;
		out = 0.0f;
		while (*in != '\0')
		{
			if (idx == 0 && *in == '-')
			{
				sign = -1;
			}
			else if (*in == '.' && comma == 0)
			{
				comma = 1;
			}
			else if (*in && *in >= '0' && *in <= '9')
			{
				out *= 10;
				out += *in - '0';
				if (comma)
				{
					divisor *= 10;
				}
			}
			else if (*in == 'f' && in[1] == '\0')
			{

			}
			else
			{
				return false;
			}

			++in;
			++idx;
		}
		out *= sign;
		out /= divisor;
		return true;
	}
	bool FromCString(const char* in, const char*& out)
	{
		out = in;
		return true;
	}
	bool FromCString(const char* in, bool& out)
	{
		if (*in == '0' || !strcmp(in, "false"))
		{
			out = false;
			return true;
		}
		else if (*in == '1' || !strcmp(in, "true"))
		{
			out = true;
			return true;
		}
		return false;
	}

	std::string IntToString(int val)
	{
		return std::to_string(val);
	}
	std::string FloatToString(float val)
	{
		return std::to_string(val);
	}
	std::string CStrToString(char const* val)
	{
		return val;
	}
	std::string BoolToString(bool val)
	{
		return val ? "true" : "false";
	}

	std::vector<std::string> SplitString(const std::string& text, char delimeter)
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