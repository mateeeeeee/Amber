#pragma once
#include <string>

namespace amber
{
	std::string GetParentPath(std::string_view complete_path);
	std::string GetFilename(std::string_view complete_path);
	std::string GetFilenameWithoutExtension(std::string_view complete_path);
	bool FileExists(std::string_view file_path);
	std::string GetExtension(std::string_view path);
}