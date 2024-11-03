#include "FilesUtil.h"
#include <filesystem>

namespace fs = std::filesystem;

namespace amber
{
	std::string GetParentPath(std::string_view complete_path)
	{
		fs::path p(complete_path);
		return p.parent_path().string();
	}
	std::string GetFilename(std::string_view complete_path)
	{
		fs::path p(complete_path);
		return p.filename().string();
	}
	std::string GetFilenameWithoutExtension(std::string_view complete_path)
	{
		fs::path p(complete_path);
		return p.filename().replace_extension().string();
	}
	Bool FileExists(std::string_view file_path)
	{
		fs::path p(file_path);
		return fs::exists(p);
	}
	std::string GetExtension(std::string_view path)
	{
		fs::path p(path);
		return p.extension().string();
		fs::file_time_type tt;
		auto x = tt.time_since_epoch().count();
	}
}

