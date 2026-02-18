#pragma once
#include <span>
#include "BVH.h"
#include "SpatialTraits.h"

// Parallel Locally-Ordered Clustering (PLOC) BVH builder
// Meister & Bittner, "Parallel Locally-Ordered Clustering for Bounding Volume Hierarchy Construction"

namespace amber
{
	class PLOCBuilder
	{
	public:
		static constexpr Int DEFAULT_RADIUS = 14;

		template<typename NodeT>
		void Build(BVH& bvh, std::span<NodeT> primitives, Int radius = DEFAULT_RADIUS);
	};
}
