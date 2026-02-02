#include "CpuPathTracer.h"
#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include "Core/Log.h"
#include "Utilities/ImageUtil.h"
#include "ImGui/imgui.h"
#include <random>

namespace amber
{
	CpuPathTracer::CpuPathTracer(Uint32 width, Uint32 height, PathTracerConfig const& config, std::unique_ptr<Scene>&& _scene)
		: width(width), height(height), scene(std::move(_scene)), framebuffer(height, width)
	{
		framebuffer.Clear(RGBA8(0, 0, 0, 255));
		AMBER_INFO_LOG("CPU PathTracer initialized");
	}

	CpuPathTracer::~CpuPathTracer()
	{
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

		std::mt19937 rng(frame_index);
		std::uniform_int_distribution<Uint32> dist(0, 255);

		for (Uint32 y = 0; y < height; ++y)
		{
			for (Uint32 x = 0; x < width; ++x)
			{
				framebuffer(y, x) = RGBA8(
					static_cast<Uint8>(dist(rng)),
					static_cast<Uint8>(dist(rng)),
					static_cast<Uint8>(dist(rng)),
					255
				);
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
	}

	void CpuPathTracer::LightsGUI()
	{
		ImGui::Text("Lights: %llu", static_cast<unsigned long long>(scene->lights.size()));
	}

	void CpuPathTracer::MemoryUsageGUI()
	{
		Uint64 fbMem = width * height * sizeof(RGBA8);
		ImGui::Text("Framebuffer: %.2f MB", fbMem / (1024.0 * 1024.0));
	}
}
