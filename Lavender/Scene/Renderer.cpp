#include "cuda_runtime.h"
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
	namespace
	{
		LAV_CUDA_KERNEL void render_kernel(Vector4* output, uint64 width, uint64 height, Camera const& camera)
		{
			//uint64 globalRow = blockIdx.y * blockDim.y + threadIdx.y;
			//uint64 globalCol = blockIdx.x * blockDim.x + threadIdx.x;
			//
			//if (globalRow < height && globalCol < width) 
			//{
			//	uint64 globalIndex = globalRow * width + globalCol;
			//
			//	output[globalIndex].x = 1.0f;
			//	output[globalIndex].y = 0.0f;
			//	output[globalIndex].z = 0.0f;
			//	output[globalIndex].w = 1.0f;
			//}
		}
	}

	Renderer::Renderer(uint32 width, uint32 height, std::unique_ptr<Scene>&& scene) 
		: framebuffer(height, width), dev_memory(width * height)
	{
		int const device = 0;
		CudaCheck(cudaSetDevice(device));
		cudaDeviceProp props{};
		CudaCheck(cudaGetDeviceProperties(&props, device));
		LAV_INFO("Device: {}\n", props.name);

		cudaMemset(dev_memory, 0, dev_memory.GetAllocSize());
	}

	Renderer::~Renderer()
	{
		CudaCheck(cudaDeviceReset());
	}

	void Renderer::Update(float dt)
	{

	}

	void Renderer::Render(Camera const& camera)
	{
		//static RealRandomGenerator rng(0.0f, 1.0f);
		//for (uint32 i = 0; i < framebuffer.Rows(); ++i)
		//{
		//	for (uint32 j = 0; j < framebuffer.Cols(); ++j)
		//	{
		//		framebuffer(i, j) = Vector4(rng(), rng(), rng(), 1.0f);
		//	}
		//}

		static constexpr uint64 BLOCK_DIM = 16;

		uint64 const width  = framebuffer.Cols();
		uint64 const height = framebuffer.Rows();
		uint64 const grid_width = (width + BLOCK_DIM - 1) / BLOCK_DIM;
		uint64 const grid_height = (height + BLOCK_DIM - 1) / BLOCK_DIM;
		dim3 const block_dim(BLOCK_DIM, BLOCK_DIM);
		dim3 const grid_dim(grid_width, grid_height);

		CUDA_LAUNCH(render_kernel, grid_dim, block_dim, dev_memory.As(), width, height, camera);
		cudaMemcpy(framebuffer, dev_memory, dev_memory.GetAllocSize(), cudaMemcpyDeviceToHost);
	}

	void Renderer::OnResize(uint32 w, uint32 h)
	{
		framebuffer.Resize(h, w);
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

