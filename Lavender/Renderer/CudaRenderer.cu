#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include "Scene.h"
#include "CudaRenderer.h"
#include "Camera.h"
#include "Core/Logger.h"
#include "Core/Paths.h"
#include "Cuda/CudaCore.h"
#include "Cuda/CudaKernel.h"
#include "Cuda/CudaMath.h"
#include "Utilities/Random.h"

//move to some utility file : ImageWrite
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "External/stb/stb_image_write.h"


namespace lavender
{
	static constexpr uint64 BLOCK_DIM = 16;

	namespace
	{
		LAV_DEVICE curandState* gpu_rand_state;

		LAV_KERNEL void RenderKernel(Pixel* output, curandState* rand, uint64 width, uint64 height, Camera const& camera)
		{
			uint64 const col = blockIdx.x * blockDim.x + threadIdx.x;
			uint64 const row = blockIdx.y * blockDim.y + threadIdx.y;

			if (col >= width || row >= height) return;

			//gpu_rand_state = rand->As();

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
		framebuffer(height, width), device_memory(width * height), cuda_rand(width * height)
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

		CudaLaunchKernel(RenderKernel, grid_dim, block_dim, device_memory, cuda_rand, width, height, camera);
		CudaCheck(cudaDeviceSynchronize());
		cudaMemcpy(framebuffer, device_memory, device_memory.GetAllocSize(), cudaMemcpyDeviceToHost);
	}

	void CudaRenderer::OnResize(uint32 w, uint32 h)
	{
		framebuffer.Resize(h, w);
		device_memory.Realloc(h * w);
		cuda_rand.Realloc(h * w);
		cudaMemset(device_memory, 0, device_memory.GetAllocSize());
	}

	void CudaRenderer::WriteFramebuffer(char const* outfile)
	{
		std::string output_path = paths::ResultDir() + outfile;
		//if(output_path.ends_with("hdr")) stbi_write_hdr(output_path.c_str(), framebuffer.Cols(), framebuffer.Rows(), 4, (float*)framebuffer.Data());
		//else if(output_path.ends_with("png"))
		//{
		//	//do tonemapping first
		//}
	}

}

