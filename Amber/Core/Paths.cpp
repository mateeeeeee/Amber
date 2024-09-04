#include "Paths.h"

namespace amber
{
	std::string const paths::MainDir = AMBER_PATH"/";
	std::string const paths::SavedDir = MainDir + "Saved/";
	std::string const paths::ResourcesDir = MainDir + "Resources/";
	std::string const paths::FontsDir = ResourcesDir + "Fonts/";
	std::string const paths::IconsDir = ResourcesDir + "Icons/";
	std::string const paths::ResultDir = SavedDir + "Result/";
	std::string const paths::LogDir = SavedDir + "Log/";
	std::string const paths::ConfigDir = SavedDir + "Config/";
	std::string const paths::SceneDir = ResourcesDir + "Scenes/";
	std::string const paths::KernelsDir = AMBER_PATH"/Optix/Kernels/";

}

