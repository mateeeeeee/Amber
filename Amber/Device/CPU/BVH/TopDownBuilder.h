#pragma once
#include <optional>
#include <span>
#include "BVH.h"
#include "PrimTraits.h"

namespace amber
{
	struct SplitResult
	{
		Int   axis;
		Float pos;
	};

	template<typename PrimitiveT, typename SplitPolicyT>
	concept TopDownSplitPolicy = requires(BVH const& bvh, PrimitiveT const* prims, BVHNode const& node)
	{
		{ SplitPolicyT::FindSplit(bvh, prims, node) } -> std::same_as<std::optional<SplitResult>>;
	};

	template<typename PrimitiveT, typename SplitPolicyT> requires TopDownSplitPolicy<PrimitiveT, SplitPolicyT>
	class TopDownBuilder
	{
		using Traits = PrimTraits<PrimitiveT>;
	public:
		void Build(BVH& bvh, std::span<PrimitiveT const> prims)
		{
			Uint32 prim_count = static_cast<Uint32>(prims.size());
			if (prim_count == 0)
			{
				return;
			}

			bvh.tri_indices.resize(prim_count);
			for (Uint32 i = 0; i < prim_count; i++)
			{
				bvh.tri_indices[i] = i;
			}

			bvh.nodes.resize(prim_count * 2 - 1);
			bvh.nodes_used = 1;

			BVHNode& root = bvh.nodes[0];
			root.left_first = 0;
			root.tri_count  = prim_count;
			UpdateNodeBounds(bvh, prims.data(), 0);
			Subdivide(bvh, prims.data(), 0);
		}

	private:
		void UpdateNodeBounds(BVH& bvh, PrimitiveT const* prims, Uint32 node_idx)
		{
			BVHNode& node = bvh.nodes[node_idx];
			AABB box{};
			for (Uint32 i = 0; i < node.tri_count; i++)
			{
				Uint32 idx = bvh.tri_indices[node.left_first + i];
				Traits::GrowBounds(box, prims[idx]);
			}
			node.aabb_min = box.min;
			node.aabb_max = box.max;
		}

		void Subdivide(BVH& bvh, PrimitiveT const* prims, Uint32 node_idx)
		{
			BVHNode& node = bvh.nodes[node_idx];
			if (node.tri_count <= 2)
			{
				return;
			}

			std::optional<SplitResult> split = SplitPolicyT::FindSplit(bvh, prims, node);
			if (!split)
			{
				return;
			}

			Uint32 first_idx = node.left_first;
			Int i = static_cast<Int>(first_idx);
			Int j = i + static_cast<Int>(node.tri_count) - 1;
			while (i <= j)
			{
				Float centroid = Traits::GetCentroid(prims[bvh.tri_indices[i]], split->axis);
				if (centroid < split->pos)
				{
					i++;
				}
				else
				{
					std::swap(bvh.tri_indices[i], bvh.tri_indices[j]);
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

			UpdateNodeBounds(bvh, prims, left_child_idx);
			UpdateNodeBounds(bvh, prims, right_child_idx);

			Subdivide(bvh, prims, left_child_idx);
			Subdivide(bvh, prims, right_child_idx);
		}
	};
}
