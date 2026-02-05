#pragma once
#include "Device/BVH/BVHData.h"
#include "Device/BVH/Intersection.h"

namespace amber
{
	template<typename BuildDataT> requires BVHBuildDataType<BuildDataT>
	struct MedianSplitBuilder
	{
		static void Build(BuildDataT& data, Triangle const* triangles, Uint32 tri_count);

	private:
		static void UpdateNodeBounds(BuildDataT& data, Triangle const* triangles, Uint32 node_idx);
		static void Subdivide(BuildDataT& data, Triangle const* triangles, Uint32 node_idx);
	};
}
