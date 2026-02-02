#include "PathTracer.h"
#include "Core/Log.h"
#include "CPU/CpuPathTracer.h"

#if defined(AMBER_HAS_METAL)
	#include "Metal/MetalPathTracer.h"
#endif

#if defined(AMBER_HAS_OPTIX)
	#include "OptiX/OptixPathTracer.h"
#endif

namespace amber
{
	std::unique_ptr<PathTracerBase> CreatePathTracer(
		PathTracerBackend backend,
		Uint32 width,
		Uint32 height,
		PathTracerConfig const& config,
		std::unique_ptr<Scene>&& scene)
	{
		if (!IsBackendAvailable(backend))
		{
			AMBER_ERROR_LOG("Backend '%s' is not available on this platform", GetBackendName(backend).c_str());
			return nullptr;
		}

		switch (backend)
		{
		case PathTracerBackend::CPU:
			return std::make_unique<CpuPathTracer>(width, height, config, std::move(scene));

#if defined(AMBER_HAS_METAL)
		case PathTracerBackend::Metal:
			return std::make_unique<MetalPathTracer>(width, height, config, std::move(scene));
#endif

#if defined(AMBER_HAS_OPTIX)
		case PathTracerBackend::OptiX:
			return std::make_unique<OptixPathTracer>(width, height, config, std::move(scene));
#endif

		default:
			AMBER_ERROR_LOG("Unknown backend requested");
			return nullptr;
		}
	}

	PathTracerBackend GetDefaultBackend()
	{
#if defined(AMBER_HAS_METAL)
		return PathTracerBackend::Metal;
#elif defined(AMBER_HAS_OPTIX)
		return PathTracerBackend::OptiX;
#else
		return PathTracerBackend::CPU;
#endif
	}

	Bool IsBackendAvailable(PathTracerBackend backend)
	{
		switch (backend)
		{
		case PathTracerBackend::CPU:
			return true;

		case PathTracerBackend::Metal:
#if defined(AMBER_HAS_METAL)
			return true;
#else
			return false;
#endif

		case PathTracerBackend::OptiX:
#if defined(AMBER_HAS_OPTIX)
			return true;
#else
			return false;
#endif

		default:
			return false;
		}
	}

	std::string GetBackendName(PathTracerBackend backend)
	{
		switch (backend)
		{
		case PathTracerBackend::Metal: return "metal";
		case PathTracerBackend::OptiX: return "optix";
		case PathTracerBackend::CPU:   return "cpu";
		}
		return "unknown";
	}

	Bool ParseBackend(std::string const& str, PathTracerBackend& backend)
	{
		if (str == "default" || str == "Default")
		{
			backend = GetDefaultBackend();
			return true;
		}
		if (str == "metal" || str == "Metal")
		{
			backend = PathTracerBackend::Metal;
			return true;
		}
		if (str == "optix" || str == "OptiX" || str == "Optix")
		{
			backend = PathTracerBackend::OptiX;
			return true;
		}
		if (str == "cpu" || str == "CPU")
		{
			backend = PathTracerBackend::CPU;
			return true;
		}
		return false;
	}
}
