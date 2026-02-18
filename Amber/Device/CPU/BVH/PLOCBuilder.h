#pragma once
#include <span>
#include "BVH.h"
#include "SpatialTraits.h"

namespace amber
{
	class PLOCBuilder
	{
	public:
		static constexpr Int DEFAULT_RADIUS = 14;

		template<typename NodeT>
		void Build(BVH2& bvh, std::span<NodeT> primitives, Int radius = DEFAULT_RADIUS);
	};
}
