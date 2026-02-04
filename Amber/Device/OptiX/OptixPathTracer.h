#pragma once
#include <memory>
#include "OptixUtils.h"
#include "DeviceHostCommon.h"
#include "Device/PathTracer.h"

namespace amber
{
	struct Scene;
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

	class OptixPathTracer : public OptixInitializer, public PathTracerBase
	{
		static constexpr Uint32 MAX_DEPTH = 3;
	public:
		OptixPathTracer(Uint32 width, Uint32 height, PathTracerConfig const& config, std::unique_ptr<Scene>&& scene);
		~OptixPathTracer() override;

		void Update(Float dt) override;
		void Render(Camera const& camera) override;

		void OnResize(Uint32 w, Uint32 h) override;
		void WriteFramebuffer(Char const* outfile) override;

		CpuBuffer2D<RGBA8> const& GetFramebuffer() const override { return framebuffer; }
		Uint32 GetMaxDepth() const override { return MAX_DEPTH; }

		void SetOutput(PathTracerOutput pto) override
		{
			output = pto;
		}
		PathTracerOutput GetOutput() const override { return output; }

		PathTracerBackend GetBackend() const override { return PathTracerBackend::OptiX; }
		Scene const& GetScene() const override { return *scene; }

		Uint   GetFrameIndex() const override { return frame_index; }
		Uint   GetTriangleCount() const override { return triangle_count; }
		Uint64 GetMemoryUsage() const override;

		Bool  SupportsAccumulation() const override { return true; }
		Bool  GetAccumulate() const override { return accumulate; }
		void  SetAccumulate(Bool v) override { accumulate = v; }

		Int   GetSampleCount() const override { return sample_count; }
		void  SetSampleCount(Int v) override { sample_count = v; }

		Int   GetDepthCount() const override { return depth_count; }
		void  SetDepthCount(Int v) override { depth_count = v; }

		Bool HasLightEditor() const override { return true; }
		void LightEditorGUI() override;

		Bool HasDenoiser() const override { return true; }
		void DenoiserGUI() override;

	private:
		Uint32 width;
		Uint32 height;
		std::unique_ptr<Scene>		scene;
		optix::TBuffer<Float3>		accum_buffer;
		optix::TBuffer<Float3>		debug_buffer;
		optix::TBuffer<Float3>		hdr_buffer;
		optix::TBuffer<Uchar4>		ldr_buffer;
		CpuBuffer2D<RGBA8>			framebuffer;

		std::unique_ptr<optix::Pipeline> pipeline;
		optix::ShaderBindingTable sbt;
		std::vector<OptixTraversableHandle> blas_handles;
		OptixTraversableHandle tlas_handle;
		std::vector<std::unique_ptr<optix::Buffer>> as_outputs;

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

		Bool	denoise = false;
		Int	    denoise_accumulation_target = 12;
		Float	denoise_blend_factor = 0.0f;
		std::unique_ptr<optix::Buffer> denoiser_state_buffer;
		std::unique_ptr<optix::Buffer> denoiser_scratch_buffer;
		optix::TBuffer<float3> denoiser_output;
		optix::TBuffer<float3> denoiser_albedo;
		optix::TBuffer<float3> denoiser_normals;
		OptixImage2D input_image;
		OptixImage2D input_albedo;
		OptixImage2D input_normals;
		OptixImage2D output_image;
		OptixImage2D debug_image;

		Bool   accumulate	= true;
		Uint   frame_index	= 0;
		Uint   triangle_count = 0;
		Int depth_count;
		Int sample_count;
		PathTracerOutput output = PathTracerOutput::Final;

	private:
		void ManageDenoiserResources();
	};
}
