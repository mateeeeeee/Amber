#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include "Scene.h"
#include "Renderer.h"
#include "Camera.h"
#include "Core/Logger.h"
#include "Core/Paths.h"
#include "Cuda/CudaUtil.h"
#include "Cuda/CudaKernel.h"
#include "Cuda/CudaMath.h"
#include "Utilities/Random.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "External/stb/stb_image_write.h"



namespace lavender
{
	namespace kernel
	{
		static constexpr uint64 BLOCK_DIM = 16;

		LAV_DEVICE curandState* rand_state;
		LAV_KERNEL void InitRandomState(curandState* _state, uint64 width, uint64 height)
		{
			uint32 col = blockIdx.x * blockDim.x + threadIdx.x;
			uint32 row = blockIdx.y * blockDim.y + threadIdx.y;
			if (col >= width || row >= height) return;
			uint64 j = row * width + col;
			rand_state = _state;
			curand_init(1984, j, 0, &rand_state[j]);
		}
		LAV_DEVICE Vec3 RandomVector(curandState* local_rand_state)
		{
			return Vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state));
		}

		//LAV_DEVICE static float ColorToLuminance(Vec3 color)
		//{
		//	return Dot(color, Vec3(0.2126f, 0.7152f, 0.0722f));
		//}
		//LAV_DEVICE static Vec3 ReinhardToneMapping(Vec3 color)
		//{
		//	float luma = ColorToLuminance(color);
		//	float toneMappedLuma = luma / (1. + luma);
		//	if (luma > 1e-6) color = color * toneMappedLuma / luma;
		//	static const float gamma = 2.2;
		//	color.x = pow(color.x, 1. / gamma);
		//	color.y = pow(color.y, 1. / gamma);
		//	color.z = pow(color.z, 1. / gamma);
		//	return color;
		//}

		LAV_KERNEL void Render(Pixel* output, uint64 width, uint64 height, Camera const& camera)
		{
			uint64 const col = blockIdx.x * blockDim.x + threadIdx.x;
			uint64 const row = blockIdx.y * blockDim.y + threadIdx.y;

			if (col >= width || row >= height) return;

			uint64 j = row * width + col;

			/*
			if (row < height / 2)
			{
				if (col < width / 2)
				{
					output[j].r = 0xff;
					output[j].g = 0;
					output[j].b = 0;
					output[j].a = 0xff;
				}
				else
				{
					output[j].r = 0;
					output[j].g = 0xff;
					output[j].b = 0;
					output[j].a = 0xff;
				}
			}
			else
			{
				if (col < width / 2)
				{
					output[j].r = 0;
					output[j].g = 0;
					output[j].b = 0xff;
					output[j].a = 0xff;
				}
				else
				{
					output[j].r = 0xff;
					output[j].g = 0xff;
					output[j].b = 0;
					output[j].a = 0xff;
				}
			}
			*/

			Vec3 pixel_color = RandomVector(&rand_state[j]);

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


	Renderer::Renderer(uint32 width, uint32 height, std::unique_ptr<Scene>&& scene)  : CudaInitializer(), 
		framebuffer(height, width), device_memory(width * height)
	{
		OnResize(width, height);
	}

	Renderer::~Renderer()
	{
	}

	void Renderer::Update(float dt)
	{
	}

	void Renderer::Render(Camera const& camera)
	{
		uint64 const width  = framebuffer.Cols();
		uint64 const height = framebuffer.Rows();

		uint64 const grid_width = (width + kernel::BLOCK_DIM - 1) / kernel::BLOCK_DIM;
		uint64 const grid_height = (height + kernel::BLOCK_DIM - 1) / kernel::BLOCK_DIM;
		dim3 const block_dim(kernel::BLOCK_DIM, kernel::BLOCK_DIM);
		dim3 const grid_dim(grid_width, grid_height);

		CudaLaunchKernel(kernel::Render, grid_dim, block_dim, device_memory.As(), width, height, camera);
		CudaCheck(cudaDeviceSynchronize());
		cudaMemcpy(framebuffer, device_memory, device_memory.GetAllocSize(), cudaMemcpyDeviceToHost);
	}

	void Renderer::OnResize(uint32 w, uint32 h)
	{
		framebuffer.Resize(h, w);
		device_memory.Realloc(h * w);
		cudaMemset(device_memory, 0, device_memory.GetAllocSize());
		curandState* _rand_state;
		CudaCheck(cudaMalloc((void**)&_rand_state, framebuffer.Size() * sizeof(curandState)));
		dim3 block(kernel::BLOCK_DIM, kernel::BLOCK_DIM);
		dim3 grid((uint32)std::ceil(w * 1.0f / block.x), (uint32)std::ceil(h * 1.0f / block.y));
		CudaLaunchKernel(kernel::InitRandomState, grid, block, _rand_state, w, h);
		CudaCheck(cudaDeviceSynchronize());
	}

	void Renderer::WriteFramebuffer(char const* outfile)
	{
		std::string output_path = paths::ResultDir() + outfile;
		//if(output_path.ends_with("hdr")) stbi_write_hdr(output_path.c_str(), framebuffer.Cols(), framebuffer.Rows(), 4, (float*)framebuffer.Data());
		//else if(output_path.ends_with("png"))
		//{
		//	//do tonemapping first
		//}
	}

}

