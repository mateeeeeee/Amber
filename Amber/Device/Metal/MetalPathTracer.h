#pragma once
#include <memory>
#include <vector>
#include "Device/PathTracer.h"

namespace amber
{
	struct Scene;
	class Camera;

	struct LightGPU;

	namespace metal
	{
		class Device;
		class Buffer;
		class Texture2D;
		class ComputePipeline;
		class AccelerationStructure;
	}

	class MetalPathTracer : public PathTracerBase
	{
		static constexpr Uint32 MAX_DEPTH = 3;
	public:
		MetalPathTracer(Uint32 width, Uint32 height, PathTracerConfig const& config, std::unique_ptr<Scene>&& scene);
		~MetalPathTracer() override;

		void Update(Float dt) override;
		void Render(Camera const& camera) override;

		void OnResize(Uint32 w, Uint32 h) override;
		void WriteFramebuffer(Char const* outfile) override;

		CpuBuffer2D<RGBA8> const& GetFramebuffer() const override { return framebuffer; }
		Uint32 GetMaxDepth() const override { return MAX_DEPTH; }

		void SetOutput(PathTracerOutput pto) override { output = pto; frame_index = 0; }
		PathTracerOutput GetOutput() const override { return output; }

		PathTracerBackend GetBackend() const override { return PathTracerBackend::Metal; }
		Scene const& GetScene() const override { return *scene; }

		Uint   GetFrameIndex() const override { return frame_index; }
		Uint   GetTriangleCount() const override;
		Uint64 GetMemoryUsage() const override { return 0; }

		Bool  SupportsAccumulation() const override { return true; }
		Bool  GetAccumulate() const override { return accumulate; }
		void  SetAccumulate(Bool v) override { accumulate = v; }

		Int   GetSampleCount() const override { return sample_count; }
		void  SetSampleCount(Int v) override { sample_count = v; }

		Int   GetDepthCount() const override { return depth_count; }
		void  SetDepthCount(Int v) override { depth_count = v; }

		Bool HasPostProcessing() const override { return true; }
		void PostProcessingGUI() override;

<<<<<<< HEAD
=======
		Bool HasLightEditor() const override { return true; }
		void LightEditorGUI() override;

>>>>>>> bvh-benchmark
	private:
		Uint32 width;
		Uint32 height;
		std::unique_ptr<Scene>		scene;
		CpuBuffer2D<RGBA8>			framebuffer;

		std::unique_ptr<metal::Device> device;

		std::unique_ptr<metal::Buffer> vertices_buffer;
		std::unique_ptr<metal::Buffer> normals_buffer;
		std::unique_ptr<metal::Buffer> uvs_buffer;
		std::unique_ptr<metal::Buffer> indices_buffer;
		std::unique_ptr<metal::Buffer> mesh_list_buffer;
		std::unique_ptr<metal::Buffer> instance_data_buffer;

		std::unique_ptr<metal::Buffer> material_list_buffer;
		std::unique_ptr<metal::Buffer> light_list_buffer;
		std::vector<LightGPU> lights;

		std::unique_ptr<metal::Texture2D> sky_texture;
		std::vector<std::unique_ptr<metal::Texture2D>> textures;

		std::unique_ptr<metal::Texture2D> accum_texture;
		std::unique_ptr<metal::Texture2D> output_texture;
		std::unique_ptr<metal::Texture2D> debug_texture;

		std::vector<std::unique_ptr<metal::AccelerationStructure>> blas_list;
		std::unique_ptr<metal::AccelerationStructure> tlas;

		std::unique_ptr<metal::ComputePipeline> pathtracer_pipeline;
		std::unique_ptr<metal::ComputePipeline> postprocess_pipeline;
		std::unique_ptr<metal::ComputePipeline> debugview_pipeline;
		std::unique_ptr<metal::Buffer> scene_argument_buffer;

		Bool   accumulate    = true;
		Uint   frame_index   = 0;
		Uint   triangle_count = 0;
		Int    depth_count;
		Int    sample_count;
		PathTracerOutput output = PathTracerOutput::Final;

		Float  exposure     = 1.0f;
		Int    tonemap_mode = 1;
	};
}
