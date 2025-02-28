#include "CommandLineOptions.h"
#include "Utilities/CLIParser.h"

namespace amber::CommandLineOptions
{
	namespace
	{
		std::string config_file;
		std::string log_file;
		std::string output_file;
		Bool use_editor = true;
		Bool maximize_window = false;
		Bool stats_enabled = false;

		void RegisterOptions(CLIParser& cli_parser)
		{
			cli_parser.AddArg(true, "--config-file");
			cli_parser.AddArg(true, "--log-file");
			cli_parser.AddArg(true, "--output-file");
			cli_parser.AddArg(false, "--noeditor");
			cli_parser.AddArg(false, "--max");

		}
	}

	void Initialize(Int argc, Char* argv[])
	{
		CLIParser cli_parser{};
		RegisterOptions(cli_parser);
		CLIParseResult parse_result = cli_parser.Parse(argc, argv);

		CLIParseResult cli_result = cli_parser.Parse(argc, argv);

		config_file = cli_result["--config-file"].AsStringOr("sponza.json");
		log_file = cli_result["--log-file"].AsStringOr("amber.log");
		output_file = cli_result["--output-file"].AsStringOr("output");
		use_editor = !cli_result["--noeditor"];
		maximize_window = !cli_result["--max"];
	}

	std::string const& GetLogFile()
	{
		return log_file;
	}
	std::string const& GetOutputFile()
	{
		return output_file;
	}
	std::string const& GetConfigFile()
	{
		return config_file;
	}
	Bool GetUseEditor()
	{
		return use_editor;
	}
	Bool GetMaximizeWindow()
	{
		return maximize_window;
	}
	Bool GetStatsEnabled()
	{
		return stats_enabled;
	}
}
