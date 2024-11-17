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
		OptixDenoiser optix_denoiser = nullptr;
	};

	class OptixRenderer : public OptixInitializer
	{
		static constexpr Uint32 MAX_DEPTH = 3;
	public:
		OptixRenderer(Uint32 width, Uint32 height, std::unique_ptr<Scene>&& scene);
		~OptixRenderer();

		void Update(Float dt);
		void Render(Camera const& camera);

		void OnResize(Uint32 w, Uint32 h);
		void WriteFramebuffer(Char const* outfile);

		auto const& GetFramebuffer() const { return framebuffer; }
		Uint32 GetMaxDepth() const { return MAX_DEPTH; }

		void OptionsGUI();
		void LightsGUI();
		void MemoryUsageGUI();

		void SetDepthCount(Uint32 depth)
		{
			depth_count = depth;
			if (depth_count > MAX_DEPTH) depth_count = MAX_DEPTH;
		}
		void SetSampleCount(Uint32 samples)
		{
			sample_count = samples;
		}

	private:
		Uint32 width;
		Uint32 height;
		std::unique_ptr<Scene>		scene;
		optix::TBuffer<float3>		accum_buffer;
		optix::TBuffer<float3>		ldr_buffer;
		optix::TBuffer<uchar4>		uchar4_ldr_buffer;
		CpuBuffer2D<uchar4>			framebuffer;

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

		Bool denoising = false;
		Sint32 denoising_accumulation_target;
		std::unique_ptr<optix::Buffer> denoiser_state_buffer;
		std::unique_ptr<optix::Buffer> denoiser_scratch_buffer;
		optix::TBuffer<float3> denoiser_output;
		optix::TBuffer<float3> denoiser_albedo;
		optix::TBuffer<float3> denoiser_normals;
		OptixImage2D input_image;
		OptixImage2D input_albedo;
		OptixImage2D input_normals;
		OptixImage2D output_image;

		Uint32 frame_index;
		Sint32 depth_count;
		Sint32 sample_count;

	private:
		void ManageDenoiserResources();
	};
}