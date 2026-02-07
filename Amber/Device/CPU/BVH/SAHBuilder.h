#pragma once
#include "BVH.h"

namespace amber
{
	class SAHBuilder
	{
	public:
		void Build(BVH& bvh, std::vector<Triangle> const& triangles);

	private:
		static constexpr Int NUM_BINS = 8;

		void UpdateNodeBounds(BVH& bvh, Triangle const* triangles, Uint32 node_idx);
		void Subdivide(BVH& bvh, Triangle const* triangles, Uint32 node_idx);
	};
}
