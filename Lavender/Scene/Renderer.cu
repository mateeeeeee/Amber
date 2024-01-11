#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Scene.h"
#include "Renderer.h"
#include "Camera.h"
#include "Core/Logger.h"
#include "Core/Paths.h"
#include "Cuda/CudaUtil.h"
#include "Cuda/CudaKernel.h"
#include "Utilities/Random.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "External/stb/stb_image_write.h"



namespace lavender
{
	static constexpr uint64 BLOCK_DIM = 16;

	__global__ void render_kernel(Pixel* output, uint64 width, uint64 height, Camera const& camera, float t)
	{
		uint64 const col = blockIdx.x * blockDim.x + threadIdx.x;
		uint64 const row = blockIdx.y * blockDim.y + threadIdx.y;

		if (col >= width || row >= height) return;

		uint64 j = row * width + col;
		output[j].r = sin(t); // uint8(sin(t) * 0xff);
		output[j].g = 0;
		output[j].b = 0;
		output[j].a = 0xff;
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
		cudaMemset(device_memory, 0, device_memory.GetAllocSize());
	}

	Renderer::~Renderer()
	{
	}

	void Renderer::Update(float dt)
	{
		time += dt;
	}

	void Renderer::Render(Camera const& camera)
	{
		uint64 const width  = framebuffer.Cols();
		uint64 const height = framebuffer.Rows();
		uint64 const grid_width = (width + BLOCK_DIM - 1) / BLOCK_DIM;
		uint64 const grid_height = (height + BLOCK_DIM - 1) / BLOCK_DIM;
		dim3 const block_dim(BLOCK_DIM, BLOCK_DIM);
		dim3 const grid_dim(grid_width, grid_height);

		CudaLaunchKernel(lavender::render_kernel, grid_dim, block_dim, device_memory.As(), width, height, camera, time);
		CudaCheck(cudaDeviceSynchronize());
		cudaMemcpy(framebuffer, device_memory, device_memory.GetAllocSize(), cudaMemcpyDeviceToHost);

		//lavender::render_kernel<<<grid_dim, block_dim>>>(device_memory.As(), width, height, camera);
		//CudaCheck(CudaLaunchKernel_Debug(lavender::render_kernel, grid_dim, block_dim, device_memory.As(), width, height, camera));
		//CudaCheck(CudaLaunchKernel_Debug(lavender::test_kernel, grid_dim, block_dim, device_memory.As()));
	}

	void Renderer::OnResize(uint32 w, uint32 h)
	{
		framebuffer.Resize(h, w);
		device_memory.Realloc(h * w);
	}

	void Renderer::WriteFramebuffer(char const* outfile)
	{
		std::string output_path = paths::ResultDir() + outfile;
		if(output_path.ends_with("hdr")) stbi_write_hdr(output_path.c_str(), framebuffer.Cols(), framebuffer.Rows(), 4, (float*)framebuffer.Data());
		else if(output_path.ends_with("png"))
		{
			//do tonemapping first
		}
	}

}

