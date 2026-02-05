#pragma once
#include <memory>
#include "Device/PathTracer.h"
#include "CpuBVH.h"
#include "Utilities/ThreadPool.h"

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

		Uint   GetFrameIndex() const override { return frame_index; }
		Uint   GetTriangleCount() const override { return static_cast<Uint>(triangles.size()); }
		Uint64 GetMemoryUsage() const override
		{
			return static_cast<Uint64>(width) * height * sizeof(RGBA8) + triangles.size() * sizeof(Triangle);
		}

	private:
		Uint32 width;
		Uint32 height;
		std::unique_ptr<Scene> scene;
		CpuBuffer2D<RGBA8> framebuffer;

		std::vector<Triangle> triangles;
		CpuBVH bvh;

		Uint frame_index = 0;
		PathTracerOutput output = PathTracerOutput::Final;

	private:
		void BuildSceneGeometry();
	};
}
