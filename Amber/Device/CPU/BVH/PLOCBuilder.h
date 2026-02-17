#pragma once
#include <span>
#include "BVH.h"
#include "SpatialTraits.h"

// Parallel Locally-Ordered Clustering (PLOC) BVH builder
// Meister & Bittner, "Parallel Locally-Ordered Clustering for Bounding Volume Hierarchy Construction"

namespace amber
{
	template<typename NodeT>
	class PLOCBuilder
	{
		using Traits = SpatialTraits<NodeT>;
	public:
		static constexpr Int DEFAULT_RADIUS = 14;

		void Build(BVH& bvh, std::span<NodeT const> primitives, Int radius = DEFAULT_RADIUS);
	};

	struct Triangle;
	struct BLASInstance;

	using PLOCBuilderBLAS = PLOCBuilder<Triangle>;
	using PLOCBuilderTLAS = PLOCBuilder<BLASInstance>;
}
