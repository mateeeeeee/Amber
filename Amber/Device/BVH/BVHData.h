#pragma once
#include <concepts>

namespace amber
{
	struct BVHNode
	{
		Vector3 aabb_min;
		Vector3 aabb_max;
		Uint32 left_first;
		Uint32 tri_count;

		Bool IsLeaf() const { return tri_count > 0; }
	};

	template<typename T>
	concept BVHBuildDataType = requires(T data, Uint32 idx)
	{
		{ data.nodes[idx] } -> std::same_as<BVHNode&>;
		{ data.tri_indices[idx] } -> std::same_as<Uint32&>;
		{ data.nodes_used } -> std::convertible_to<Uint32&>;
	};
}
