#pragma once
#include <memory>
#include <unordered_map>
#include "Device/PathTracer.h"
#include "Device/CPU/AccelerationStructure.h"
#include "Device/CPU/Texture.h"
#include "Scene/Light.h"
#include "Utilities/ThreadPool.h"
#include "Utilities/Timer.h"
#include "Scene/Mesh.h"
#include "Device/CPU/BVH/Stats.h"

namespace amber
{
	struct Scene;
	class Camera;

	class CpuPathTracer : public PathTracerBase
	{

	public:
		CpuPathTracer(Uint32 width, Uint32 height, PathTracerConfig const& config, std::unique_ptr<Scene>&& scene);
		~CpuPathTracer() override;

		void Update(Float dt) override;
		void Render(Camera const& camera) override;

		void OnResize(Uint32 w, Uint32 h) override;
		void WriteFramebuffer(Char const* outfile) override;

		CpuBuffer2D<RGBA8> const& GetFramebuffer() const override { return framebuffer; }
		Uint32 GetMaxDepth() const override { return max_depth; }

		void SetOutput(PathTracerOutput pto) override { output = pto; }
		PathTracerOutput GetOutput() const override { return output; }

		PathTracerBackend GetBackend() const override { return PathTracerBackend::CPU; }
		Scene const& GetScene() const override { return *scene; }

		Float  GetRenderTime() const override { return render_time_ms; }
		Uint   GetFrameIndex() const override { return frame_index; }
		Uint   GetTriangleCount() const override { return triangle_count; }
		Uint64 GetMemoryUsage() const override { return 0; }

		Bool HasPostProcessing() const override { return true; }
		void PostProcessingGUI() override;

		Bool HasLightEditor() const override { return true; }
		void LightEditorGUI() override;

		Bool HasBVHDebug() const override { return true; }
		void BVHDebugGUI() override;

	private:
		Uint32 width;
		Uint32 height;
		Uint32 max_depth;
		Uint32 tile_size;
		std::unique_ptr<Scene> scene;
		CpuBuffer2D<RGBA8>    framebuffer;
		CpuBuffer2D<Vector3>  accumulation_buffer;

		std::vector<BLAS>              blas_list;
		std::vector<Geometry const*>   blas_geometries;
		std::vector<Uint32>            blas_material_ids;
		std::vector<Texture>           textures;
		Texture                        env_texture;
		TLAS                           tlas;
		std::vector<Light>             lights;
		Uint triangle_count = 0;

		Uint frame_index = 0;
		Float render_time_ms = 0.0f;
		PathTracerOutput output = PathTracerOutput::Final;

		Float exposure     = 1.0f;
		Int   tonemap_mode = 1;

		Bool  bvh_heatmap_enabled    = false;
		Bool  bvh_heatmap_blend		 = false;
		Int   bvh_heatmap_max_steps  = 64;
		Int   bvh_heatmap_mode       = 0; // add enum for this later
		Float bvh_heatmap_blend_alpha = 0.5f;
		Int   bvh_selected_blas      = -1;  
		BVHStats bvh_tlas_stats{};
		std::vector<BVHStats> bvh_blas_stats;

	private:
		void BuildAccelerationStructures();
	};
}
