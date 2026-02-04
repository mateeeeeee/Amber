#pragma once
#include <memory>
#include "Device/PathTracer.h"
#include "BVH.h"
#include "Utilities/ThreadPool.h"

namespace amber
{
	class Scene;
	class Camera;

	class CpuPathTracer : public PathTracerBase
	{
		static constexpr Uint32 MAX_DEPTH = 3;
		static constexpr Uint32 TILE_SIZE = 16;
		
	public:
		CpuPathTracer(Uint32 width, Uint32 height, PathTracerConfig const& config, std::unique_ptr<Scene>&& scene);
		~CpuPathTracer() override;

		void Update(Float dt) override;
		void Render(Camera const& camera) override;

		void OnResize(Uint32 w, Uint32 h) override;
		void WriteFramebuffer(Char const* outfile) override;

		CpuBuffer2D<RGBA8> const& GetFramebuffer() const override { return framebuffer; }
		Uint32 GetMaxDepth() const override { return MAX_DEPTH; }

		void SetOutput(PathTracerOutput pto) override { output = pto; }
		PathTracerOutput GetOutput() const override { return output; }

		void OptionsGUI() override;
		void LightsGUI() override;
		void MemoryUsageGUI() override;

		PathTracerBackend GetBackend() const override { return PathTracerBackend::CPU; }

	private:
		Uint32 width;
		Uint32 height;
		std::unique_ptr<Scene> scene;
		CpuBuffer2D<RGBA8> framebuffer;

		std::vector<Triangle> triangles;
		BVH bvh;

		Uint frame_index = 0;
		PathTracerOutput output = PathTracerOutput::Final;

	private:
		void BuildSceneGeometry();
	};
}
