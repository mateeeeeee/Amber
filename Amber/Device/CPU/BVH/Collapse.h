#pragma once
#include "BVH.h"

namespace amber
{
	namespace detail
	{
		template<Uint32 N>
		static Uint32 CollectChildren(BVH2 const& bvh2, Uint32 bvh2_idx, Uint32 out_children[N])
		{
			BVH2Node const& node = bvh2.nodes[bvh2_idx];
			Uint32 open[N];
			Uint32 open_count = 0;
			open[open_count++] = node.children[0];
			open[open_count++] = node.children[1];

			while (open_count < N)
			{
				Int expand = -1;
				for (Uint32 i = 0; i < open_count; i++)
				{
					if (!bvh2.nodes[open[i]].IsLeaf())
					{
						expand = static_cast<Int>(i);
						break;
					}
				}

				if (expand < 0) 
				{
					break;
				} 

				BVH2Node const& to_expand = bvh2.nodes[open[expand]];
				for (Uint32 i = open_count; i > static_cast<Uint32>(expand) + 1; i--)
				{
					open[i] = open[i - 1];
				}
				open[expand]     = to_expand.children[0];
				open[expand + 1] = to_expand.children[1];
				open_count++;
			}

			for (Uint32 i = 0; i < open_count; i++) out_children[i] = open[i];
			return open_count;
		}

		template<Uint32 N>
		static Uint32 CollapseRecursive(BVH2 const& bvh2, BVH<N>& bvhn, Uint32 bvh2_idx)
		{
			BVH2Node const& src = bvh2.nodes[bvh2_idx];

			Uint32 dst_idx = bvhn.nodes_used++;
			BVHNode<N>& dst = bvhn.nodes[dst_idx];
			dst.aabb_min = src.aabb_min;
			dst.aabb_max = src.aabb_max;

			if (src.IsLeaf())
			{
				dst.child_count = 0;
				dst.first_prim  = src.first_prim;
				dst.prim_count  = src.prim_count;
				return dst_idx;
			}

			Uint32 bvh2_children[N];
			Uint32 child_count = CollectChildren<N>(bvh2, bvh2_idx, bvh2_children);
			dst.child_count = child_count;

			for (Uint32 i = 0; i < child_count; i++)
			{
				Uint32 child_dst = CollapseRecursive<N>(bvh2, bvhn, bvh2_children[i]);
				bvhn.nodes[dst_idx].children[i] = child_dst;
			}

			return dst_idx;
		}
	}

	template<Uint32 N>
	void Collapse(BVH2 const& bvh2, BVH<N>& bvhn)
	{
		static_assert(N == 4 || N == 8);

		if (bvh2.nodes_used == 0)
		{
			bvhn.nodes_used = 0;
			return;
		}

		bvhn.nodes.resize(bvh2.nodes_used);
		bvhn.prim_indices = bvh2.prim_indices;
		bvhn.nodes_used   = 0;
		detail::CollapseRecursive<N>(bvh2, bvhn, 0);
	}
}
