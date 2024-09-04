#include <nvrtc.h>
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

	KernelPTX CompileKernel(KernelCompilerInput const& compiler_input)
	{
		std::string_view kernel_file = compiler_input.kernel_file;

		FILE* file = fopen(kernel_file.data(), "rb");
		fseek(file, 0, SEEK_END);
		uint64 input_size = ftell(file);
		std::unique_ptr<char[]> kernel(new char[input_size]);
		rewind(file);
		fread(kernel.get(), sizeof(char), input_size, file);
		fclose(file);

		nvrtcProgram prog;
		NvrtcCheck(nvrtcCreateProgram(&prog, kernel.get(), kernel_file.data(), 0, nullptr, nullptr));

		const char* options[] = 
		{
		"--pre-include=Core/CoreTypes.h",
		"--pre-include=Core/Defines.h",
		"--pre-include=Math/MathTypes.h",
		"-I" AMBER_PATH,
		"-I\"C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0/include\"",
		"-I\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include\"",
		"--use_fast_math",
		"--generate-line-info",
		"-std=c++17"
		};

		nvrtcResult compile_result = nvrtcCompileProgram(prog, ARRAYSIZE(options), options);
		if (compile_result != NVRTC_SUCCESS) 
		{
			uint64 log_size;
			nvrtcGetProgramLogSize(prog, &log_size);
			std::string log(log_size, '\0');
			nvrtcGetProgramLog(prog, &log[0]);
			AMBER_ERROR("Compilation failed: %s", log.c_str());
			nvrtcDestroyProgram(&prog);
			return {};
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

