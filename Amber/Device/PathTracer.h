#pragma once
#include <memory>
#include <string>
#include "Utilities/CpuBuffer2D.h"

namespace amber
{
	struct Scene;
	class Camera;

	enum class PathTracerBackend : Uint8
	{
		Metal,
		OptiX,
		CPU
	};

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

	class PathTracerBase
	{
	public:
		virtual ~PathTracerBase() = default;

		virtual void Update(Float dt) = 0;
		virtual void Render(Camera const& camera) = 0;
		virtual void OnResize(Uint32 w, Uint32 h) = 0;
		virtual void WriteFramebuffer(Char const* outfile) = 0;

		virtual CpuBuffer2D<RGBA8> const& GetFramebuffer() const = 0;
		virtual Uint32 GetMaxDepth() const = 0;

		virtual void SetOutput(PathTracerOutput pto) = 0;
		virtual PathTracerOutput GetOutput() const = 0;

		virtual PathTracerBackend GetBackend() const = 0;
		virtual Scene const& GetScene() const = 0;

		virtual Float  GetRenderTime() const { return 0.0f; }
		virtual Uint   GetFrameIndex() const = 0;
		virtual Uint   GetTriangleCount() const { return 0; }
		virtual Uint64 GetMemoryUsage() const { return 0; }

		virtual Bool  SupportsAccumulation() const { return false; }
		virtual Bool  GetAccumulate() const { return false; }
		virtual void  SetAccumulate(Bool) {}

		virtual Int   GetSampleCount() const { return 1; }
		virtual void  SetSampleCount(Int) {}

		virtual Int   GetDepthCount() const { return 1; }
		virtual void  SetDepthCount(Int) {}

		virtual Bool HasLightEditor() const { return false; }
		virtual void LightEditorGUI() {}

		virtual Bool HasBVHDebug() const { return false; }
		virtual void BVHDebugGUI() {}

		virtual Bool HasDenoiser() const { return false; }
		virtual void DenoiserGUI() {}

		virtual Bool HasPostProcessing() const { return false; }
		virtual void PostProcessingGUI() {}
	};

	std::unique_ptr<PathTracerBase> CreatePathTracer(
		PathTracerBackend backend,
		Uint32 width,
		Uint32 height,
		PathTracerConfig const& config,
		std::unique_ptr<Scene>&& scene);

	PathTracerBackend GetDefaultBackend();
	Bool IsBackendAvailable(PathTracerBackend backend);
	std::string GetBackendName(PathTracerBackend backend);
	Bool ParseBackend(std::string const& str, PathTracerBackend& backend);
}
