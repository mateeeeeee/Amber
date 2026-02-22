#pragma once
#include <cstdlib>
#include <string>
#include <type_traits>

namespace amber
{
	Char const* GetEnvVar(Char const* name);

	template<typename T>
	T GetEnvVar(char const* name, T default_value)
	{
		Char const* val = std::getenv(name);
		if (!val) 
		{
			return default_value;
		}

		if constexpr (std::is_same_v<T, Int>)
		{
			return std::atoi(val);
		}
		else if constexpr (std::is_same_v<T, Float>)
		{
			return std::strtof(val, nullptr);
		}
		else if constexpr (std::is_same_v<T, std::string>)
		{
			return std::string(val);
		}
		else
		{
			static_assert(!sizeof(T), "GetEnvVar: unsupported type T");
		}
	}
}
