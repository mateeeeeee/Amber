#pragma once
#include <string_view>
#include <vector>

namespace amber
{
	struct KernelDefine
	{
		std::string name;
		std::string value;
	};

	struct KernelCompilerInput
	{
		std::string_view kernel_file;
		std::vector<KernelDefine> defines;
	};

	using KernelPTX = std::vector<Char>;
	Bool CompileKernel(KernelCompilerInput const& compiler_input, KernelPTX& ptx);
}