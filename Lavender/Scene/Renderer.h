#pragma once
#include <memory>
#include "Cuda/CudaAlloc.h"
#include "Cuda/CudaEvent.h"
#include "Utilities/Buffer2D.h"
#include "Core/Defines.h"
#include "Math/MathTypes.h"

namespace lavender
{
	struct Pixel
	{
		uint8 r, g, b, a;
	};

	class Scene;
	class Camera;
	using Framebuffer = Buffer2D<Pixel>;
	using DeviceMemory = TypedCudaAlloc<Pixel>;

	class CudaInitializer
	{
	public:
		CudaInitializer();
		~CudaInitializer();
	};

	class Renderer : public CudaInitializer
	{
	public:
		explicit Renderer(uint32 width, uint32 height, std::unique_ptr<Scene>&& scene);
		~Renderer();

		void Update(float dt);
		void Render(Camera const& camera);

		void OnResize(uint32 w, uint32 h);
		void WriteFramebuffer(char const* outfile);
		
		Framebuffer const& GetFramebuffer() const { return framebuffer; }

	private:
		Framebuffer   framebuffer;
		DeviceMemory  device_memory;

	private:
	};
}