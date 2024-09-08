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

	bool ReadFileContents(std::string_view file_path, std::vector<char>& buffer)
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

		std::vector<char> kernel;
		if (!ReadFileContents(kernel_file, kernel))
		{
			return std::unexpected(CompilerError::ReadFileFailed);
		}
		
		nvrtcProgram prog;
		NvrtcCheck(nvrtcCreateProgram(&prog, kernel.data(), kernel_file.data(), 0, nullptr, nullptr));

		std::string optix_include = "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0/include";
		std::string cuda_include = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include";
		std::string cuda_std_include = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include/cuda/std";
		optix_include = "-I" + optix_include;
		cuda_include = "-I" + cuda_include;
		cuda_std_include = "-I" + cuda_std_include;

		std::vector<char const*> compile_options;
		compile_options.push_back(optix_include.c_str());
		compile_options.push_back(cuda_include.c_str());
		compile_options.push_back(cuda_std_include.c_str());
		compile_options.push_back("-I" AMBER_PATH);
		compile_options.push_back("--pre-include=Core/CoreTypes.h");
		compile_options.push_back("--use_fast_math");
		compile_options.push_back("--generate-line-info");

		nvrtcResult compile_result = nvrtcCompileProgram(prog, compile_options.size(), compile_options.data());
		if (compile_result != NVRTC_SUCCESS) 
		{
			uint64 log_size;
			nvrtcGetProgramLogSize(prog, &log_size);
			std::string log(log_size, '\0');
			nvrtcGetProgramLog(prog, &log[0]);
			AMBER_ERROR("Compilation failed: %s", log.c_str());
			nvrtcDestroyProgram(&prog);
			return std::unexpected(CompilerError::CompilationFailed);
		}

		uint64 ptx_size;
		nvrtcGetPTXSize(prog, &ptx_size);
		
		KernelPTX ptx{};
		ptx.resize(ptx_size);
		nvrtcGetPTX(prog, ptx.data());

		nvrtcDestroyProgram(&prog);
		return ptx;
	}

}

