#include "cuda_runtime.h"
#include "Scene.h"
#include "Renderer.h"
#include "Camera.h"
#include "Core/Logger.h"
#include "Cuda/CudaUtil.h"

namespace lavender
{

	Renderer::Renderer(std::unique_ptr<Scene>&& scene) : scene(std::move(scene))
	{
		CudaCheck(cudaSetDevice(0));
	}

	Renderer::~Renderer()
	{
		CudaCheck(cudaDeviceReset());
	}

	void Renderer::Update(float dt)
	{

	}

	void Renderer::Render()
	{

	}

}

