#include "cuda_runtime.h"
#include "Scene.h"
#include "Renderer.h"
#include "Camera.h"
#include "Core/Logger.h"
#include "Core/Paths.h"
#include "Cuda/CudaUtil.h"
#include "Utilities/Random.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "External/stb/stb_image_write.h"


namespace lavender
{
	Renderer::Renderer(uint32 width, uint32 height, std::unique_ptr<Scene>&& scene) 
		: framebuffer(height, width)
	{
		int const device = 0;
		CudaCheck(cudaSetDevice(device));
		cudaDeviceProp props{};
		CudaCheck(cudaGetDeviceProperties(&props, device));
		LAV_INFO("Device: {}\n", props.name);

		RealRandomGenerator rng(0.0f, 1.0f);
		for (uint32 i = 0; i < framebuffer.Rows(); ++i)
		{
			for (uint32 j = 0; j < framebuffer.Cols(); ++j)
			{
				framebuffer(i, j) = Vector4(rng(), rng(), rng(), 1.0f);
			}
		}
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

	}

	void Renderer::OnResize(uint32 w, uint32 h)
	{
		framebuffer.Resize(h, w);
	}

	void Renderer::WriteFramebuffer(char const* outfile)
	{
		std::string output_path = paths::ResultDir() + outfile;
		if(output_path.ends_with("hdr")) stbi_write_hdr(output_path.c_str(), framebuffer.Cols(), framebuffer.Rows(), 4, (float*)framebuffer.Data());
		else
		{

		}
	}

}

