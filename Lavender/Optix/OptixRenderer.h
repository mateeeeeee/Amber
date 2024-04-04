#pragma once
#include <memory>
#include "OptixUtils.h"
#include "Utilities/CpuBuffer2D.h"

namespace lavender
{
	class Scene;
	class Camera;
	using Framebuffer = CpuBuffer2D<uchar4>;
}

namespace lavender::optix
{
	using DeviceMemory = optix::TypedBuffer<uchar4>;

	class OptixInitializer
	{
	public:
		OptixInitializer();
		~OptixInitializer();

	protected:
		CUcontext cuda_context = nullptr;
		OptixDeviceContext optix_context = nullptr;
	};

	class OptixRenderer : public OptixInitializer
	{
	public:
		OptixRenderer(uint32 width, uint32 height, std::unique_ptr<Scene>&& scene);
		~OptixRenderer();

		void Update(float dt);
		void Render(Camera const& camera);

		void OnResize(uint32 w, uint32 h);
		void WriteFramebuffer(char const* outfile);
		
		Framebuffer const& GetFramebuffer() const { return framebuffer; }

	private:
		Framebuffer   framebuffer;
		DeviceMemory  device_memory;

		std::unique_ptr<Pipeline> pipeline;
		ShaderBindingTable sbt;
		OptixTraversableHandle blas_handle;
	};
}