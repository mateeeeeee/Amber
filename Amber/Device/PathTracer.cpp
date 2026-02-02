#include "PathTracer.h"
#include "Core/Log.h"

#if defined(__APPLE__)
	#include "Metal/MetalPathTracer.h"
#else
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

#if defined(__APPLE__)
		return std::make_unique<MetalPathTracer>(width, height, config, std::move(scene));
#else
		return std::make_unique<OptixPathTracer>(width, height, config, std::move(scene));
#endif
	}

	PathTracerBackend GetDefaultBackend()
	{
#if defined(__APPLE__)
		return PathTracerBackend::Metal;
#else
		return PathTracerBackend::OptiX;
#endif
	}

	Bool IsBackendAvailable(PathTracerBackend backend)
	{
#if defined(__APPLE__)
		return backend == PathTracerBackend::Metal;
#else
		return backend == PathTracerBackend::OptiX;
#endif
	}

	std::string GetBackendName(PathTracerBackend backend)
	{
		switch (backend)
		{
		case PathTracerBackend::Metal: return "metal";
		case PathTracerBackend::OptiX: return "optix";
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
		return false;
	}
}
