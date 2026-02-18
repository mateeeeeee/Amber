#include "PLOCBuilder.h"
#include "Device/CPU/TLAS.h"
#include <algorithm>
#include <numeric>
#include <vector>

namespace amber
{
	namespace 
	{
		inline constexpr Uint32 ExpandBits(Uint32 v)
		{
			v &= 0x000003ffu;
			v = (v ^ (v << 16u)) & 0xff0000ffu;
			v = (v ^ (v <<  8u)) & 0x0300f00fu;
			v = (v ^ (v <<  4u)) & 0x030c30c3u;
			v = (v ^ (v <<  2u)) & 0x09249249u;
			return v;
		}

		inline constexpr Uint32 MortonCode3D(Float x, Float y, Float z)
		{
			x = std::clamp(x * 1024.0f, 0.0f, 1023.0f);
			y = std::clamp(y * 1024.0f, 0.0f, 1023.0f);
			z = std::clamp(z * 1024.0f, 0.0f, 1023.0f);
			return (ExpandBits(static_cast<Uint32>(x)) << 2u) |
			       (ExpandBits(static_cast<Uint32>(y)) << 1u) |
			        ExpandBits(static_cast<Uint32>(z));
		}
	}

	template<typename NodeT>
	void PLOCBuilder::Build(BVH& bvh, std::span<NodeT> nodes, Int radius)
	{
		using Traits = SpatialTraits<NodeT>;

		Uint32 n = static_cast<Uint32>(nodes.size());
		if (n == 0) 
		{
			return;
		}

		// Worst case scenario, we have n leaves + up to (n-1) merges * 3 = 4n-3 nodes (each merge = 3 nodes)
		// This is not ideal, look into improving this
		// todo check this: https://madmann91.github.io/2020/12/28/bvhs-part-1.html 
		bvh.nodes.resize(4 * n);
		bvh.prim_indices.resize(n);
		bvh.nodes_used = 0;

		// Calculate the scene centroid needed for morton code normalization
		AABB centroid_aabb{};
		for (Uint32 i = 0; i < n; i++)
		{
			Vector3 c(Traits::GetCentroid(nodes[i], 0),
			          Traits::GetCentroid(nodes[i], 1),
			          Traits::GetCentroid(nodes[i], 2));
			centroid_aabb.Grow(c);
		}

		Vector3 extent = centroid_aabb.max - centroid_aabb.min;
		if (extent.x == 0.0f) extent.x = 1.0f;
		if (extent.y == 0.0f) extent.y = 1.0f;
		if (extent.z == 0.0f) extent.z = 1.0f;

		std::vector<Uint32> order(n);
		std::iota(order.begin(), order.end(), 0u);
		std::vector<Uint32> morton(n);
		for (Uint32 i = 0; i < n; i++)
		{
			Float cx = (Traits::GetCentroid(nodes[i], 0) - centroid_aabb.min.x) / extent.x;
			Float cy = (Traits::GetCentroid(nodes[i], 1) - centroid_aabb.min.y) / extent.y;
			Float cz = (Traits::GetCentroid(nodes[i], 2) - centroid_aabb.min.z) / extent.z;
			morton[i] = MortonCode3D(cx, cy, cz);
		}

		std::sort(order.begin(), order.end(), [&](Uint32 a, Uint32 b)
		{
			return morton[a] < morton[b];
		});

		for (Uint32 i = 0; i < n; i++) 
		{
			bvh.prim_indices[i] = order[i];
		}

		std::vector<Uint32> clusters(n);
		for (Uint32 i = 0; i < n; i++)
		{
			Uint32 idx      = bvh.nodes_used++;
			clusters[i]     = idx;

			BVHNode& leaf   = bvh.nodes[idx];
			leaf.left_first = i;
			leaf.prim_count = 1;

			AABB box{};
			Traits::GrowBounds(box, nodes[order[i]]);
			leaf.aabb_min = box.min;
			leaf.aabb_max = box.max;
		}

		std::vector<Int>    nearest(n);
		std::vector<Uint32> next_clusters;
		next_clusters.reserve(n);
		std::vector<Bool>   valid(n);
		Uint32 cluster_count = n;
		while (cluster_count > 1)
		{
			nearest.resize(cluster_count);
			valid.assign(cluster_count, true);
			next_clusters.clear();

			for (Uint32 i = 0; i < cluster_count; i++)
			{
				BVHNode const& ni = bvh.nodes[clusters[i]];
				AABB box_i(ni.aabb_min, ni.aabb_max);

				Float best_area = BVH_INFINITY;
				Int   best_j    = -1;

				Int lo = std::max(0, static_cast<Int>(i) - radius);
				Int hi = std::min(static_cast<Int>(cluster_count) - 1, static_cast<Int>(i) + radius);

				// Search interval [i - r, i + r] for the nearest neighbour using distance function d
				/*From paper: "We define a distance function d between two clusters C1
				and C2 as the surface area A of an axis aligned bounding
				box tightly enclosing C1 and C2" 
				*/
				for (Int j = lo; j <= hi; j++)
				{
					if (static_cast<Uint32>(j) == i) 
					{
						continue;
					}

					BVHNode const& nj = bvh.nodes[clusters[j]];
					AABB merged = box_i;
					merged.Grow(AABB(nj.aabb_min, nj.aabb_max));
					Float area = merged.Area();

					/* From paper: "To guarantee that the algorithm always terminates, we prioritize the nearest
									neighbor with the lower index which solves a potential rare
									case of completely equidistant clusters" */
					if (area < best_area || (area == best_area && j < best_j))
					{
						best_area = area;
						best_j    = j;
					}
				}
				nearest[i] = best_j;
			}

			for (Uint32 i = 0; i < cluster_count; i++)
			{
				Int j = nearest[i];
				if (j < 0) 
				{
					continue;
				}

				// "To avoid conflicts, merging is performed by a thread processing the cluster with the lower index."
				if (nearest[j] == static_cast<Int>(i) && i < static_cast<Uint32>(j))
				{
					Uint32 left_node  = clusters[i];
					Uint32 right_node = clusters[j];

					Uint32 parent_idx = bvh.nodes_used++;
					Uint32 left_dst   = bvh.nodes_used++;
					Uint32 right_dst  = bvh.nodes_used++;

					bvh.nodes[left_dst]  = bvh.nodes[left_node];
					bvh.nodes[right_dst] = bvh.nodes[right_node];

					BVHNode& parent = bvh.nodes[parent_idx];
					AABB merged(bvh.nodes[left_dst].aabb_min,  bvh.nodes[left_dst].aabb_max);
					merged.Grow(AABB(bvh.nodes[right_dst].aabb_min, bvh.nodes[right_dst].aabb_max));
					parent.aabb_min   = merged.min;
					parent.aabb_max   = merged.max;
					parent.left_first = left_dst;
					parent.prim_count = 0;

					clusters[i] = parent_idx;
					valid[j]    = false;
				}
			}

			for (Uint32 i = 0; i < cluster_count; i++)
			{
				if (valid[i]) 
				{
					next_clusters.push_back(clusters[i]);
				}
			}

			clusters.swap(next_clusters);
			cluster_count = static_cast<Uint32>(clusters.size());
		}

		Uint32 root_src = clusters[0];
		if (root_src != 0)
		{
			bvh.nodes[0] = bvh.nodes[root_src];
		}
	}

	template void PLOCBuilder::Build(BVH&, std::span<Triangle>,     Int);
	template void PLOCBuilder::Build(BVH&, std::span<BLASInstance>, Int);
}
