#pragma once
#include <memory>
#include "Utilities/CpuBuffer2D.h"

namespace amber
{
	class Scene;
	class Camera;

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

	// Metal-based path tracer for macOS
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

		Bool   accumulate	= true;
		Uint   frame_index	= 0;
		Int depth_count;
		Int sample_count;
		PathTracerOutput output = PathTracerOutput::Final;
	};
}
