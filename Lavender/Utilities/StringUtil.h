#pragma once
#include <string>
#include <vector>

namespace lavender
{
	std::string ToLower(std::string const& in);
	std::string ToUpper(std::string const& in);

	bool FromCString(char const* in, int& out);
	bool FromCString(char const* in, float& out);
	bool FromCString(char const* in, const char*& out);
	bool FromCString(char const* in, bool& out);

	std::string IntToString(int val);
	std::string FloatToString(float val);
	std::string CStrToString(char const* val);
	std::string BoolToString(bool val);

	std::vector<std::string> SplitString(std::string const& text, char delimeter);
}