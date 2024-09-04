#pragma once
#include <string_view>
#include <vector>

namespace amber
{
	struct KernelCompilerInput
	{
		std::string_view kernel_file;
		//add defines?
	};

	using KernelPTX = std::vector<char>;
	KernelPTX CompileKernel(KernelCompilerInput const& compiler_input);
}