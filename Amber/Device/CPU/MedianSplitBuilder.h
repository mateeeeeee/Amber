#pragma once
#include <vector>
#include "BVH.h"

namespace amber
{
	struct MedianSplitBuilder
	{
		static void Build(BVHBuildData& data, std::vector<Triangle> const& triangles);

	private:
		static void UpdateNodeBounds(BVHBuildData& data, std::vector<Triangle> const& triangles, Uint32 node_idx);
		static void Subdivide(BVHBuildData& data, std::vector<Triangle> const& triangles, Uint32 node_idx);
	};
}
