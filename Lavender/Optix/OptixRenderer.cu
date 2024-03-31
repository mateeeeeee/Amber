#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include "CudaUtils.h"
#include "MathUtils.h"
#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include "OptixRenderer.h"
#include "Core/Logger.h"
#include "Core/Paths.h"
#include "Utilities/Random.h"
#include "Utilities/ImageUtil.h"


namespace lavender::optix
{
	static constexpr uint64 BLOCK_DIM = 16;

	static void OptixLogCallback(unsigned int level, const char* tag, const char* message, void* cbdata)
	{
		switch (level)
		{
		case 0:
		case 1:
		case 2:
		case 3:
			return;
		}
	}

	OptixInitializer::OptixInitializer()
	{
		int num_devices = 0;
		cudaGetDeviceCount(&num_devices);
		if (num_devices == 0) 
		{
			LAV_ERROR("No CUDA devices found!");
			std::exit(1);
		}

		OptixCheck(optixInit());

		int const device = 0;
		CudaCheck(cudaSetDevice(device));
		cudaDeviceProp props{};
		CudaCheck(cudaGetDeviceProperties(&props, device));
		LAV_INFO("Device: %s\n", props.name);

		cuCtxGetCurrent(&cuda_context);

		OptixCheck(optixDeviceContextCreate(cuda_context, nullptr, &optix_context));
		OptixCheck(optixDeviceContextSetLogCallback(optix_context, OptixLogCallback, nullptr, 0));
	}

	OptixInitializer::~OptixInitializer()
	{
		OptixCheck(optixDeviceContextDestroy(optix_context));
		CudaCheck(cudaDeviceReset());
	}


	OptixRenderer::OptixRenderer(uint32 width, uint32 height, std::unique_ptr<Scene>&& scene)  : OptixInitializer(), 
		framebuffer(height, width), device_memory(width * height)
	{
		OnResize(width, height);
	}

	OptixRenderer::~OptixRenderer()
	{
	}

	void OptixRenderer::Update(float dt)
	{
	}

	void OptixRenderer::Render(Camera const& camera)
	{
		uint64 const width  = framebuffer.Cols();
		uint64 const height = framebuffer.Rows();

		uint64 const grid_width = (width + BLOCK_DIM - 1) / BLOCK_DIM;
		uint64 const grid_height = (height + BLOCK_DIM - 1) / BLOCK_DIM;
		dim3 const block_dim(BLOCK_DIM, BLOCK_DIM);
		dim3 const grid_dim(grid_width, grid_height);
	}

	void OptixRenderer::OnResize(uint32 w, uint32 h)
	{
		framebuffer.Resize(h, w);
		if (device_memory.GetCount() != w * h)
		{
			device_memory = optix::TypedBuffer<Pixel>(w * h);
		}
		cudaMemset(device_memory, 0, device_memory.GetSize());
	}

	void OptixRenderer::WriteFramebuffer(char const* outfile)
	{
		std::string output_path = paths::ResultDir() + outfile;
		WriteImageToFile(ImageFormat::PNG, output_path.data(), framebuffer.Cols(), framebuffer.Rows(), framebuffer.Data(), sizeof(Pixel));
	}

}

