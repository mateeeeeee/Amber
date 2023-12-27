#pragma once
#include <string_view>

namespace lavender
{
	struct PbrtFileLocation
	{
		std::string filename = "";
		uint32 line = 1;
		uint32 column = 1;

		PbrtFileLocation operator+(int32 i)
		{
			return PbrtFileLocation
			{
				.filename = filename,
				.line = line,
				.column = column + i
			};
		}

		void NewChar()
		{
			++column;
		}
		void NewChars(int32 i)
		{
			column += i;
		}
		void NewLine()
		{
			++line;
			column = 1;
		}
	};
}