#pragma once
#include <string_view>
#include <vector>
#include <expected>

namespace amber
{
	struct KernelCompilerInput
	{
		std::string_view kernel_file;
		//add defines?
	};

	enum class CompilerError
	{
		ReadFileFailed,
		CompilationFailed
	};

	using KernelPTX = std::vector<char>;
	std::expected<KernelPTX, CompilerError> CompileKernel(KernelCompilerInput const& compiler_input);
}