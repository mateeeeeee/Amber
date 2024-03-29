#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include "CudaRenderer.h"
#include "Cuda/CudaCore.h"
#include "Cuda/CudaKernel.h"
#include "Cuda/CudaMath.h"
#include "Core/Logger.h"
#include "Core/Paths.h"
#include "Utilities/Random.h"
#include "Utilities/ImageUtil.h"


namespace lavender
{
	static constexpr uint64 BLOCK_DIM = 16;

	namespace
	{
		LAV_KERNEL void DeviceRandInit(curandState* rand_state, uint64 size)
		{
			uint64 j = blockIdx.x * blockDim.x + threadIdx.x;
			if (j >= size) return;
			curand_init(1984, j, 0, &rand_state[j]);
		}
		LAV_KERNEL void RenderKernel(Pixel* output, curandState* rand, uint64 width, uint64 height, Camera const& camera)
		{
			uint64 const col = blockIdx.x * blockDim.x + threadIdx.x;
			uint64 const row = blockIdx.y * blockDim.y + threadIdx.y;

			if (col >= width || row >= height) return;

			uint64 j = row * width + col;
			Vec3 pixel_color(curand_uniform(&rand[j]), curand_uniform(&rand[j]), curand_uniform(&rand[j]));

			output[j].r = static_cast<uint8>(pixel_color.x * 0xff);
			output[j].g = static_cast<uint8>(pixel_color.y * 0xff);
			output[j].b = static_cast<uint8>(pixel_color.z * 0xff);
			output[j].a = 0xff;
		}
	}
	
	CudaInitializer::CudaInitializer()
	{
		int const device = 0;
		CudaCheck(cudaSetDevice(device));
		cudaDeviceProp props{};
		CudaCheck(cudaGetDeviceProperties(&props, device));
		LAV_INFO("Device: %s\n", props.name);
	}

	CudaInitializer::~CudaInitializer()
	{
		CudaCheck(cudaDeviceReset());
	}


	CudaRenderer::CudaRenderer(uint32 width, uint32 height, std::unique_ptr<Scene>&& scene)  : CudaInitializer(), 
		framebuffer(height, width), device_memory(width * height), device_rand(width * height)
	{
		OnResize(width, height);
	}

	CudaRenderer::~CudaRenderer()
	{
	}

	void CudaRenderer::Update(float dt)
	{
	}

	void CudaRenderer::Render(Camera const& camera)
	{
		uint64 const width  = framebuffer.Cols();
		uint64 const height = framebuffer.Rows();

		uint64 const grid_width = (width + BLOCK_DIM - 1) / BLOCK_DIM;
		uint64 const grid_height = (height + BLOCK_DIM - 1) / BLOCK_DIM;
		dim3 const block_dim(BLOCK_DIM, BLOCK_DIM);
		dim3 const grid_dim(grid_width, grid_height);

		CudaLaunchKernel(RenderKernel, grid_dim, block_dim, device_memory, device_rand, width, height, camera);
		cudaMemcpy(framebuffer, device_memory, device_memory.GetAllocSize(), cudaMemcpyDeviceToHost);
	}

	void CudaRenderer::OnResize(uint32 w, uint32 h)
	{
		framebuffer.Resize(h, w);
		device_memory.Realloc(h * w);
		device_rand.Realloc(h * w);
		cudaMemset(device_memory, 0, device_memory.GetAllocSize());

		dim3 block(BLOCK_DIM, 1, 1);
		dim3 grid((uint32)std::ceil(device_rand.GetCount() / BLOCK_DIM), 1, 1);
		CudaLaunchKernel(DeviceRandInit, grid, block, device_rand, device_rand.GetCount());
	}

	void CudaRenderer::WriteFramebuffer(char const* outfile)
	{
		std::string output_path = paths::ResultDir() + outfile;
		WriteImageToFile(ImageFormat::PNG, output_path.data(), framebuffer.Cols(), framebuffer.Rows(), framebuffer.Data(), sizeof(Pixel));
	}

}

