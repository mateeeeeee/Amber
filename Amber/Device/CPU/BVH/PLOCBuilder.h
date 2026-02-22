#pragma once
#include <span>
#include "BVH.h"
#include "SpatialTraits.h"

namespace amber
{
	class PLOCBuilder
	{
	public:
		PLOCBuilder();

		template<typename NodeT>
		void Build(BVH2& bvh, std::span<NodeT> primitives);

	private:
		Int radius;
	};
}
