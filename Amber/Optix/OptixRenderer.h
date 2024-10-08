#pragma once
#include <memory>
#include "OptixUtils.h"
#include "OptixShared.h"
#include "Utilities/CpuBuffer2D.h"

namespace amber
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
		static constexpr uint32 MAX_DEPTH = 3;
	public:
		OptixRenderer(uint32 width, uint32 height, std::unique_ptr<Scene>&& scene);
		~OptixRenderer();

		void Update(float dt);
		void Render(Camera const& camera);

		void OnResize(uint32 w, uint32 h);
		void WriteFramebuffer(char const* outfile);

		auto const& GetFramebuffer() const { return framebuffer; }
		uint32 GetMaxDepth() const { return MAX_DEPTH; }
		void GUI();

		void SetDepthCount(uint32 depth)
		{
			depth_count = depth;
			if (depth_count > MAX_DEPTH) depth_count = MAX_DEPTH;
		}
		void SetSampleCount(uint32 samples)
		{
			sample_count = samples;
		}

	private:
		std::unique_ptr<Scene>		scene;
		CpuBuffer2D<uchar4>			framebuffer;
		optix::TBuffer<uchar4>		device_memory;
		optix::TBuffer<float4>		accum_memory;

		std::unique_ptr<optix::Pipeline> pipeline;
		optix::ShaderBindingTable sbt;
		std::vector<OptixTraversableHandle> blas_handles;
		OptixTraversableHandle tlas_handle;
		std::vector<std::unique_ptr<optix::Buffer>> as_outputs; //is it necessary to keep this alive?

		std::unique_ptr<optix::Texture2D> sky_texture;
		std::vector<std::unique_ptr<optix::Texture2D>> textures;
		std::unique_ptr<optix::Buffer> texture_list_buffer;
		std::unique_ptr<optix::Buffer> material_list_buffer;
		std::unique_ptr<optix::Buffer> light_list_buffer;
		std::unique_ptr<optix::Buffer> mesh_list_buffer;
		std::unique_ptr<optix::Buffer> vertices_buffer;
		std::unique_ptr<optix::Buffer> normals_buffer;
		std::unique_ptr<optix::Buffer> uvs_buffer;
		std::unique_ptr<optix::Buffer> indices_buffer;

		std::vector<LightGPU> lights;

		uint32 frame_index;
		int32 depth_count;
		int32 sample_count;
	};
}