#include "CpuPathTracer.h"
#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include "Core/Log.h"
#include "Utilities/ImageUtil.h"
#include "ImGui/imgui.h"
#include <cmath>

namespace amber
{
	CpuPathTracer::CpuPathTracer(Uint32 width, Uint32 height, PathTracerConfig const& config, std::unique_ptr<Scene>&& _scene)
		: width(width), height(height), scene(std::move(_scene)), framebuffer(height, width)
	{
		framebuffer.Clear(RGBA8(0, 0, 0, 255));
		BuildSceneGeometry();
		bvh.Build<MedianSplitBuilder>(triangles);
		AMBER_INFO_LOG("CPU PathTracer initialized with %zu triangles", triangles.size());
	}

	CpuPathTracer::~CpuPathTracer()
	{
	}

	void CpuPathTracer::BuildSceneGeometry()
	{
		triangles.clear();

		for (Instance const& instance : scene->instances)
		{
			Mesh const& mesh = scene->meshes[instance.mesh_id];
			Matrix const& transform = instance.transform;

			for (Geometry const& geom : mesh.geometries)
			{
				for (Vector3u const& idx : geom.indices)
				{
					Triangle tri;
					tri.v0 = Vector3::Transform(geom.vertices[idx.x], transform);
					tri.v1 = Vector3::Transform(geom.vertices[idx.y], transform);
					tri.v2 = Vector3::Transform(geom.vertices[idx.z], transform);
					tri.centroid = (tri.v0 + tri.v1 + tri.v2) * (1.0f / 3.0f);
					triangles.push_back(tri);
				}
			}
		}
	}

	void CpuPathTracer::Update(Float dt)
	{
	}

	void CpuPathTracer::Render(Camera const& camera)
	{
		if (camera.IsChanged())
		{
			frame_index = 0;
		}

		Vector3 origin = camera.GetPosition();
		Vector3 U, V, W;
		camera.GetFrame(U, V, W);

		Float fov_rad = camera.GetFovY() * (3.14159265359f / 180.0f);
		Float tan_half_fov = std::tan(fov_rad * 0.5f);
		Float aspect = camera.GetAspectRatio();
		for (Uint32 y = 0; y < height; ++y)
		{
			for (Uint32 x = 0; x < width; ++x)
			{
				Float ndc_x = (2.0f * (x + 0.5f) / width - 1.0f) * aspect * tan_half_fov;
				Float ndc_y = (1.0f - 2.0f * (y + 0.5f) / height) * tan_half_fov;

				Vector3 direction = (W + U * ndc_x + V * ndc_y).Normalized();
				Ray ray(origin, direction);

				HitInfo hit;
				if (bvh.Intersect(ray, hit))
				{
					Triangle const& tri = triangles[hit.tri_idx];
					Vector3 e1 = tri.v1 - tri.v0;
					Vector3 e2 = tri.v2 - tri.v0;
					Vector3 normal = Vector3::Cross(e1, e2).Normalized();

					RGBA8 color = RGBA8::FromFloat(
						normal.x * 0.5f + 0.5f,
						normal.y * 0.5f + 0.5f,
						normal.z * 0.5f + 0.5f
					);
					framebuffer(y, x) = color;
				}
				else
				{
					framebuffer(y, x) = RGBA8(25, 25, 25, 255);
				}
			}
		}

		++frame_index;
	}

	void CpuPathTracer::OnResize(Uint32 w, Uint32 h)
	{
		width = w;
		height = h;
		framebuffer.Resize(height, width);
		frame_index = 0;
	}

	void CpuPathTracer::WriteFramebuffer(Char const* outfile)
	{
		WriteImageToFile(ImageFormat::PNG, outfile, width, height, framebuffer.Data(), width * 4);
		AMBER_INFO_LOG("Wrote framebuffer to %s", outfile);
	}

	void CpuPathTracer::OptionsGUI()
	{
		ImGui::Text("CPU Path Tracer");
		ImGui::Separator();
		ImGui::Text("Frame: %u", frame_index);
		ImGui::Text("Triangles: %zu", triangles.size());
	}

	void CpuPathTracer::LightsGUI()
	{
		ImGui::Text("Lights: %llu", static_cast<unsigned long long>(scene->lights.size()));
	}

	void CpuPathTracer::MemoryUsageGUI()
	{
		Uint64 fbMem = width * height * sizeof(RGBA8);
		Uint64 triMem = triangles.size() * sizeof(Triangle);
		ImGui::Text("Framebuffer: %.2f MB", fbMem / (1024.0 * 1024.0));
		ImGui::Text("Triangles: %.2f MB", triMem / (1024.0 * 1024.0));
	}
}
