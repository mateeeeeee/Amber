#pragma once
#include "BVH.h"

namespace amber
{
	class MedianSplitBuilder
	{
	public:
		void Build(BVH& bvh, std::vector<Triangle> const& triangles);

	private:
		void UpdateNodeBounds(BVH& bvh, Triangle const* triangles, Uint32 node_idx);
		void Subdivide(BVH& bvh, Triangle const* triangles, Uint32 node_idx);
	};
}
