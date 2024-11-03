#pragma once
#include <string_view>
#include <vector>
#include <expected>

namespace amber
{
	struct KernelCompilerInput
	{
		std::string_view kernel_file;
		std::vector<std::string> defines;
	};

	enum class CompilerError
	{
		ReadFileFailed,
		CompilationFailed
	};

	using KernelPTX = std::vector<Char>;
	std::expected<KernelPTX, CompilerError> CompileKernel(KernelCompilerInput const& compiler_input);
}