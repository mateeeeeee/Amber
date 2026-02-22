#include "TopDownBuilder.h"
#include "SAHBuilder.h"
#include "MedianSplitBuilder.h"
#include "Utilities/EnvVars.h"
#include "Device/CPU/TLAS.h"

namespace amber
{
	static constexpr Int TOP_DOWN_DEFAULT_MAX_LEAF_PRIMS = 2;

	template<typename SplitPolicyT>
	TopDownBuilder<SplitPolicyT>::TopDownBuilder()
		: max_leaf_prims(GetEnvVar("AMBER_TOP_DOWN_MAX_LEAF_PRIMS", TOP_DOWN_DEFAULT_MAX_LEAF_PRIMS))
	{}

	template<typename SplitPolicyT>
	template<typename NodeT> requires TopDownSplitPolicy<NodeT, SplitPolicyT>
	void TopDownBuilder<SplitPolicyT>::Build(BVH2& bvh, std::span<NodeT> nodes)
	{
		Uint32 node_count = static_cast<Uint32>(nodes.size());
		if (node_count == 0)
		{
			return;
		}

		bvh.prim_indices.resize(node_count);
		for (Uint32 i = 0; i < node_count; i++)
		{
			bvh.prim_indices[i] = i;
		}

		bvh.nodes.resize(node_count * 2 - 1);
		bvh.nodes_used = 1;

		BVH2Node& root   = bvh.nodes[0];
		root.child_count = 0;
		root.first_prim  = 0;
		root.prim_count  = node_count;
		UpdateNodeBounds(bvh, nodes.data(), 0);
		Subdivide(bvh, nodes.data(), 0);
	}

	template<typename SplitPolicyT>
	template<typename NodeT>
	void TopDownBuilder<SplitPolicyT>::UpdateNodeBounds(BVH2& bvh, NodeT const* nodes, Uint32 node_idx)
	{
		using Traits = SpatialTraits<NodeT>;
		BVH2Node& node = bvh.nodes[node_idx];
		AABB box{};
		for (Uint32 i = 0; i < node.prim_count; i++)
		{
			Uint32 idx = bvh.prim_indices[node.first_prim + i];
			Traits::GrowBounds(box, nodes[idx]);
		}
		node.aabb_min = box.min;
		node.aabb_max = box.max;
	}

	template<typename SplitPolicyT>
	template<typename NodeT>
	void TopDownBuilder<SplitPolicyT>::Subdivide(BVH2& bvh, NodeT const* nodes, Uint32 node_idx)
	{
		using Traits = SpatialTraits<NodeT>;
		BVH2Node& node = bvh.nodes[node_idx];
		if (node.prim_count <= max_leaf_prims)
		{
			return;
		}

		std::optional<SplitResult> split = SplitPolicyT::FindSplit(bvh, nodes, node);
		if (!split)
		{
			return;
		}

		Uint32 first_idx = node.first_prim;
		Int i = static_cast<Int>(first_idx);
		Int j = i + static_cast<Int>(node.prim_count) - 1;
		while (i <= j)
		{
			Float centroid = Traits::GetCentroid(nodes[bvh.prim_indices[i]], split->axis);
			if (centroid < split->pos)
			{
				i++;
			}
			else
			{
				std::swap(bvh.prim_indices[i], bvh.prim_indices[j]);
				j--;
			}
		}

		Uint32 left_count = static_cast<Uint32>(i) - first_idx;
		if (left_count == 0 || left_count == node.prim_count)
		{
			return;
		}

		Uint32 left_child_idx  = bvh.nodes_used++;
		Uint32 right_child_idx = bvh.nodes_used++;

		bvh.nodes[left_child_idx].child_count  = 0;
		bvh.nodes[left_child_idx].first_prim   = first_idx;
		bvh.nodes[left_child_idx].prim_count   = left_count;
		bvh.nodes[right_child_idx].child_count = 0;
		bvh.nodes[right_child_idx].first_prim  = static_cast<Uint32>(i);
		bvh.nodes[right_child_idx].prim_count  = node.prim_count - left_count;

		node.child_count = 2;
		node.children[0] = left_child_idx;
		node.children[1] = right_child_idx;

		UpdateNodeBounds(bvh, nodes, left_child_idx);
		UpdateNodeBounds(bvh, nodes, right_child_idx);

		Subdivide(bvh, nodes, left_child_idx);
		Subdivide(bvh, nodes, right_child_idx);
	}

	template void TopDownBuilder<BinnedSAHPolicy>::Build(BVH2&, std::span<Triangle>);
	template void TopDownBuilder<BinnedSAHPolicy>::Build(BVH2&, std::span<BLASInstance>);
	template void TopDownBuilder<SweepSAHPolicy>::Build(BVH2&, std::span<Triangle>);
	template void TopDownBuilder<SweepSAHPolicy>::Build(BVH2&, std::span<BLASInstance>);
	template void TopDownBuilder<MedianSplitPolicy>::Build(BVH2&, std::span<Triangle>);
	template void TopDownBuilder<MedianSplitPolicy>::Build(BVH2&, std::span<BLASInstance>);

	template class TopDownBuilder<BinnedSAHPolicy>;
	template class TopDownBuilder<SweepSAHPolicy>;
	template class TopDownBuilder<MedianSplitPolicy>;
}
