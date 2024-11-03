#include <nvrtc.h>
#include <fstream>
#include "KernelCompiler.h"
#include "Core/Logger.h"
#include "Core/Paths.h"

namespace amber
{
    inline constexpr void NvrtcCheck(nvrtcResult result)
    {
        if (result != NVRTC_SUCCESS)
        {
            AMBER_ERROR("NVRTC ERROR: %s", nvrtcGetErrorString(result));
        }
    }

	Bool ReadFileContents(std::string_view file_path, std::vector<Char>& buffer)
	{
		std::ifstream file(file_path.data(), std::ios::ate);
		if (!file.is_open()) 
		{
			AMBER_ERROR("Failed to open file %s \n", file_path.data());
			return false;
		}
		std::streamsize file_size = file.tellg();
		buffer.resize(static_cast<std::size_t>(file_size));

		file.seekg(0, std::ios::beg);
		file.read(buffer.data(), file_size);
		return true;
	}

	std::expected<KernelPTX, CompilerError> CompileKernel(KernelCompilerInput const& compiler_input)
	{
		std::string_view kernel_file = compiler_input.kernel_file;

		std::vector<Char> kernel;
		if (!ReadFileContents(kernel_file, kernel))
		{
			return std::unexpected(CompilerError::ReadFileFailed);
		}
		
		nvrtcProgram prog;
		NvrtcCheck(nvrtcCreateProgram(&prog, kernel.data(), kernel_file.data(), 0, nullptr, nullptr));
		std::vector<Char const*> compile_options;
		compile_options.push_back("-I" OPTIX_PATH "/include");
		compile_options.push_back("-I" CUDA_PATH "/include");
		compile_options.push_back("-I" CUDA_PATH "/include/cuda/std");
		compile_options.push_back("-I" AMBER_PATH);
		compile_options.push_back("--pre-include=Core/Types.h");
		compile_options.push_back("--use_fast_math");
		compile_options.push_back("--generate-line-info");

		nvrtcResult compile_result = nvrtcCompileProgram(prog, compile_options.size(), compile_options.data());
		if (compile_result != NVRTC_SUCCESS) 
		{
			Uint64 log_size;
			nvrtcGetProgramLogSize(prog, &log_size);
			std::string log(log_size, '\0');
			nvrtcGetProgramLog(prog, &log[0]);
			AMBER_ERROR("Compilation failed: %s", log.c_str());
			nvrtcDestroyProgram(&prog);
			return std::unexpected(CompilerError::CompilationFailed);
		}

		Uint64 ptx_size;
		nvrtcGetPTXSize(prog, &ptx_size);
		
		KernelPTX ptx{};
		ptx.resize(ptx_size);
		nvrtcGetPTX(prog, ptx.data());

		nvrtcDestroyProgram(&prog);
		return ptx;
	}

}

