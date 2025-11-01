#pragma once

// Platform-specific path tracer selection
// Each platform has exactly one path tracer implementation at compile time
#if defined(AMBER_PLATFORM_APPLE)
	#include "Metal/MetalPathTracer.h"
	namespace amber
	{
		using PathTracer = MetalPathTracer;
	}
#else
	#include "OptiX/OptixPathTracer.h"
	namespace amber
	{
		using PathTracer = OptixPathTracer;
	}
#endif
