#pragma once
#include <memory>
#include <vector>
#include "Utilities/CpuBuffer2D.h"

namespace amber
{
	class Scene;
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

	struct PathTracerConfig
	{
		Uint   max_depth;
		Uint   samples_per_pixel;
		Bool   use_denoiser;
		Bool   accumulate;
	};

	enum class PathTracerOutput : Uint8
	{
		Final,
		Albedo,
		Normal,
		UV,
		MaterialID,
		Custom
	};

	class MetalPathTracer
	{
		static constexpr Uint32 MAX_DEPTH = 3;
	public:
		MetalPathTracer(Uint32 width, Uint32 height, PathTracerConfig const& config, std::unique_ptr<Scene>&& scene);
		~MetalPathTracer();

		void Update(Float dt);
		void Render(Camera const& camera);

		void OnResize(Uint32 w, Uint32 h);
		void WriteFramebuffer(Char const* outfile);

		auto const& GetFramebuffer() const { return framebuffer; }
		Uint32 GetMaxDepth() const { return MAX_DEPTH; }

		void SetOutput(PathTracerOutput pto)
		{
			output = pto;
		}
		PathTracerOutput GetOutput() const { return output; }

		void OptionsGUI();
		void LightsGUI();
		void MemoryUsageGUI();

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

		std::unique_ptr<metal::Buffer> material_list_buffer;
		std::unique_ptr<metal::Buffer> light_list_buffer;
		std::vector<LightGPU> lights;

		std::unique_ptr<metal::Texture2D> sky_texture;
		std::vector<std::unique_ptr<metal::Texture2D>> textures;

		std::unique_ptr<metal::Texture2D> accum_texture;
		std::unique_ptr<metal::Texture2D> output_texture;

		std::vector<std::unique_ptr<metal::AccelerationStructure>> blas_list;
		std::unique_ptr<metal::AccelerationStructure> tlas;

		std::unique_ptr<metal::ComputePipeline> pathtracer_pipeline;
		std::unique_ptr<metal::Buffer> scene_argument_buffer;

		Bool   accumulate	= true;
		Uint   frame_index	= 0;
		Int depth_count;
		Int sample_count;
		PathTracerOutput output = PathTracerOutput::Final;
	};
}
