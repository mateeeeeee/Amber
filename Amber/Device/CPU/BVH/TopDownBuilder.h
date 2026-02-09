#pragma once
#include <optional>
#include <span>
#include "BVH.h"
#include "SpatialTraits.h"

namespace amber
{
	struct SplitResult
	{
		Int   axis;
		Float pos;
	};

	template<typename NodeT, typename SplitPolicyT>
	concept TopDownSplitPolicy = requires(BVH const& bvh, NodeT const* nodes, BVHNode const& node)
	{
		{ SplitPolicyT::FindSplit(bvh, nodes, node) } -> std::same_as<std::optional<SplitResult>>;
	};

	template<typename NodeT, typename SplitPolicyT> requires TopDownSplitPolicy<NodeT, SplitPolicyT>
	class TopDownBuilder
	{
		using Traits = SpatialTraits<NodeT>;
	public:
		void Build(BVH& bvh, std::span<NodeT const> nodes)
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

			BVHNode& root = bvh.nodes[0];
			root.left_first = 0;
			root.tri_count  = node_count;
			UpdateNodeBounds(bvh, nodes.data(), 0);
			Subdivide(bvh, nodes.data(), 0);
		}

	private:
		void UpdateNodeBounds(BVH& bvh, NodeT const* nodes, Uint32 node_idx)
		{
			BVHNode& node = bvh.nodes[node_idx];
			AABB box{};
			for (Uint32 i = 0; i < node.tri_count; i++)
			{
				Uint32 idx = bvh.prim_indices[node.left_first + i];
				Traits::GrowBounds(box, nodes[idx]);
			}
			node.aabb_min = box.min;
			node.aabb_max = box.max;
		}

		void Subdivide(BVH& bvh, NodeT const* nodes, Uint32 node_idx)
		{
			BVHNode& node = bvh.nodes[node_idx];
			if (node.tri_count <= 2)
			{
				return;
			}

			std::optional<SplitResult> split = SplitPolicyT::FindSplit(bvh, nodes, node);
			if (!split)
			{
				return;
			}

			Uint32 first_idx = node.left_first;
			Int i = static_cast<Int>(first_idx);
			Int j = i + static_cast<Int>(node.tri_count) - 1;
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
			if (left_count == 0 || left_count == node.tri_count)
			{
				return;
			}

			Uint32 left_child_idx  = bvh.nodes_used++;
			Uint32 right_child_idx = bvh.nodes_used++;

			bvh.nodes[left_child_idx].left_first  = first_idx;
			bvh.nodes[left_child_idx].tri_count   = left_count;
			bvh.nodes[right_child_idx].left_first = static_cast<Uint32>(i);
			bvh.nodes[right_child_idx].tri_count  = node.tri_count - left_count;

			node.left_first = left_child_idx;
			node.tri_count  = 0;

			UpdateNodeBounds(bvh, nodes, left_child_idx);
			UpdateNodeBounds(bvh, nodes, right_child_idx);

			Subdivide(bvh, nodes, left_child_idx);
			Subdivide(bvh, nodes, right_child_idx);
		}
	};
}
