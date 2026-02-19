#pragma once
#include <algorithm>
#include "BVH.h"

namespace amber
{
	struct BVHStats
	{
		Uint32 node_count          = 0;
		Uint32 leaf_count          = 0;
		Uint32 internal_count      = 0;
		Uint32 max_depth           = 0;
		Uint32 nodes_only_leaves   = 0; 
		Uint32 nodes_only_internal = 0; 

		Uint32 min_leaf_prims = UINT32_MAX;
		Uint32 max_leaf_prims = 0;
		Float  avg_leaf_prims = 0.0f;

		Float  sah_cost     = 0.0f; 

		Float  total_sa      = 0.0f;
		Float  leaf_sa_total = 0.0f;
		Float  leaf_sa_avg   = 0.0f;
		Float  leaf_sa_min   = std::numeric_limits<Float>::max();
		Float  leaf_sa_max   = 0.0f;

		Float  total_volume      = 0.0f;
		Float  leaf_volume_total = 0.0f;
		Float  leaf_volume_avg   = 0.0f;
		Float  leaf_volume_min   = std::numeric_limits<Float>::max();
		Float  leaf_volume_max   = 0.0f;
	};

	template<Uint32 N>
	BVHStats ComputeStats(BVH<N> const& bvh)
	{
		BVHStats stats{};
		if (bvh.nodes_used == 0) 
		{
			return stats;
		}

		BVHNode<N> const& root_node = bvh.nodes[0];
		Vector3 re = root_node.aabb_max - root_node.aabb_min;
		Float root_area = re.x * re.y + re.y * re.z + re.z * re.x;
		if (root_area <= 0.0f) 
		{
			root_area = 1.0f;
		}
		Uint32 total_leaf_prims = 0;

		struct Entry { Uint32 idx; Uint32 depth; };
		std::vector<Entry> stack;
		stack.reserve(64);
		stack.push_back({ 0, 0 });
		while (!stack.empty())
		{
			auto [idx, depth] = stack.back();
			stack.pop_back();

			BVHNode<N> const& node = bvh.nodes[idx];
			stats.node_count++;
			stats.max_depth = std::max(stats.max_depth, depth);

			Vector3 e    = node.aabb_max - node.aabb_min;
			Float area   = e.x * e.y + e.y * e.z + e.z * e.x;
			Float volume = e.x * e.y * e.z;
			stats.total_sa     += area;
			stats.total_volume += volume;

			if (node.IsLeaf())
			{
				stats.leaf_count++;
				stats.min_leaf_prims  = std::min(stats.min_leaf_prims, node.prim_count);
				stats.max_leaf_prims  = std::max(stats.max_leaf_prims, node.prim_count);
				total_leaf_prims     += node.prim_count;

				stats.leaf_sa_total  += area;
				stats.leaf_sa_min     = std::min(stats.leaf_sa_min, area);
				stats.leaf_sa_max     = std::max(stats.leaf_sa_max, area);

				stats.leaf_volume_total += volume;
				stats.leaf_volume_min    = std::min(stats.leaf_volume_min, volume);
				stats.leaf_volume_max    = std::max(stats.leaf_volume_max, volume);

				stats.sah_cost += (area / root_area) * static_cast<Float>(node.prim_count);
			}
			else
			{
				stats.internal_count++;
				stats.sah_cost += (area / root_area);

				Bool all_leaves   = true;
				Bool all_internal = true;
				for (Uint32 i = 0; i < node.child_count; i++)
				{
					Bool child_leaf = bvh.nodes[node.children[i]].IsLeaf();
					all_leaves   &= child_leaf;
					all_internal &= !child_leaf;
					stack.push_back({ node.children[i], depth + 1 });
				}
				if (all_leaves)   
				{
					stats.nodes_only_leaves++;
				}
				if (all_internal) 
				{
					stats.nodes_only_internal++;
				}
			}
		}

		if (stats.leaf_count > 0)
		{
			stats.avg_leaf_prims   = static_cast<Float>(total_leaf_prims) / static_cast<Float>(stats.leaf_count);
			stats.leaf_sa_avg      = stats.leaf_sa_total  / static_cast<Float>(stats.leaf_count);
			stats.leaf_volume_avg  = stats.leaf_volume_total / static_cast<Float>(stats.leaf_count);
		}
		if (stats.min_leaf_prims == UINT32_MAX) 
		{
			stats.min_leaf_prims = 0;
		}
		if (stats.leaf_sa_min   == std::numeric_limits<Float>::max()) 
		{
			stats.leaf_sa_min = 0.0f;
		}
		if (stats.leaf_volume_min == std::numeric_limits<Float>::max()) 
		{
			stats.leaf_volume_min = 0.0f;
		}
		return stats;
	}
}
