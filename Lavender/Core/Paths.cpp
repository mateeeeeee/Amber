#include "Paths.h"

namespace lavender
{

	std::string paths::MainDir()
	{
		return LAVENDER_PATH"/";
	}

	std::string paths::SavedDir()
	{
		return MainDir() + "Saved/";
	}

	std::string paths::ResourcesDir()
	{
		return MainDir() + "Resources/";
	}

	std::string paths::FontsDir()
	{
		return ResourcesDir() + "Fonts/";
	}

	std::string paths::IconsDir()
	{
		return ResourcesDir() + "Icons/";
	}

	std::string paths::ResultDir()
	{
		return SavedDir() + "Result/";
	}

	std::string paths::LogDir()
	{
		return SavedDir() + "Log/";
	}

	std::string paths::ConfigDir()
	{
		return SavedDir() + "Config/";
	}

	std::string paths::SceneDir()
	{
		return ResourcesDir() + "Scenes/";
	}

}

