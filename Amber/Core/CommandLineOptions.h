#pragma once
#include <string>

namespace amber
{
	namespace CommandLineOptions
	{
		void Initialize(Int argc, Char* argv[]);
		std::string const& GetLogFile();
		std::string const& GetOutputFile();
		std::string const& GetConfigFile();
		Bool GetUseEditor();
		Bool GetMaximizeWindow();
		Bool GetStatsEnabled();
	}
}