#pragma once
#include <memory>
#include "OptixUtils.h"
#include "Utilities/CpuBuffer2D.h"

namespace lavender
{
	class Scene;
	class Camera;

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
		void Render(Camera& camera, uint32 sample_count);

		void OnResize(uint32 w, uint32 h);
		void WriteFramebuffer(char const* outfile);
		
		auto const& GetFramebuffer() const { return framebuffer; }

	private:
		std::unique_ptr<Scene>		scene;
		CpuBuffer2D<uchar4>			framebuffer;
		optix::TypedBuffer<uchar4>  device_memory;

		std::unique_ptr<optix::Pipeline> pipeline;
		optix::ShaderBindingTable sbt;
		OptixTraversableHandle as_handle;
		std::unique_ptr<optix::Buffer> as_output;

		std::vector<std::unique_ptr<optix::Texture2D>> textures;
		std::unique_ptr<optix::Buffer> texture_list_buffer;
		uint32 frame_index;
	};
}