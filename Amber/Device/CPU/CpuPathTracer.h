#pragma once
#include <memory>
#include "Device/PathTracer.h"
#include "Device/CPU/AccelerationStructure.h"
#include "Utilities/ThreadPool.h"
#include "Utilities/Timer.h"

namespace amber
{
	struct Scene;
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

		PathTracerBackend GetBackend() const override { return PathTracerBackend::CPU; }
		Scene const& GetScene() const override { return *scene; }

		Float  GetRenderTime() const override { return render_time_ms; }
		Uint   GetFrameIndex() const override { return frame_index; }
		Uint   GetTriangleCount() const override { return triangle_count; }
		Uint64 GetMemoryUsage() const override { return 0; }

	private:
		Uint32 width;
		Uint32 height;
		std::unique_ptr<Scene> scene;
		CpuBuffer2D<RGBA8> framebuffer;

		std::vector<BLAS> blas_list;
		TLAS tlas;
		Uint triangle_count = 0;

		Uint frame_index = 0;
		Float render_time_ms = 0.0f;
		PathTracerOutput output = PathTracerOutput::Final;

	private:
		void BuildAccelerationStructures();
	};
}
