#include "SAHBuilder.h"
#include <algorithm>

namespace amber
{
	void SAHBuilder::Build(BVH& bvh, std::vector<Triangle> const& triangles)
	{
		Uint32 tri_count = static_cast<Uint32>(triangles.size());
		if (tri_count == 0)
		{
			return;
		}

		bvh.triangles = &triangles;

		bvh.tri_indices.resize(tri_count);
		for (Uint32 i = 0; i < tri_count; i++)
		{
			bvh.tri_indices[i] = i;
		}

		bvh.nodes.resize(tri_count * 2 - 1);
		bvh.nodes_used = 1;

		BVHNode& root = bvh.nodes[0];
		root.left_first = 0;
		root.tri_count = tri_count;
		UpdateNodeBounds(bvh, triangles.data(), 0);
		Subdivide(bvh, triangles.data(), 0);
	}

	void SAHBuilder::UpdateNodeBounds(BVH& bvh, Triangle const* triangles, Uint32 node_idx)
	{
		BVHNode& node = bvh.nodes[node_idx];
		node.aabb_min = Vector3(1e30f, 1e30f, 1e30f);
		node.aabb_max = Vector3(-1e30f, -1e30f, -1e30f);
		for (Uint32 i = 0; i < node.tri_count; i++)
		{
			Uint32 tri_idx = bvh.tri_indices[node.left_first + i];
			Triangle const& tri = triangles[tri_idx];

			node.aabb_min.x = std::min(node.aabb_min.x, tri.v0.x);
			node.aabb_min.y = std::min(node.aabb_min.y, tri.v0.y);
			node.aabb_min.z = std::min(node.aabb_min.z, tri.v0.z);
			node.aabb_max.x = std::max(node.aabb_max.x, tri.v0.x);
			node.aabb_max.y = std::max(node.aabb_max.y, tri.v0.y);
			node.aabb_max.z = std::max(node.aabb_max.z, tri.v0.z);

			node.aabb_min.x = std::min(node.aabb_min.x, tri.v1.x);
			node.aabb_min.y = std::min(node.aabb_min.y, tri.v1.y);
			node.aabb_min.z = std::min(node.aabb_min.z, tri.v1.z);
			node.aabb_max.x = std::max(node.aabb_max.x, tri.v1.x);
			node.aabb_max.y = std::max(node.aabb_max.y, tri.v1.y);
			node.aabb_max.z = std::max(node.aabb_max.z, tri.v1.z);

			node.aabb_min.x = std::min(node.aabb_min.x, tri.v2.x);
			node.aabb_min.y = std::min(node.aabb_min.y, tri.v2.y);
			node.aabb_min.z = std::min(node.aabb_min.z, tri.v2.z);
			node.aabb_max.x = std::max(node.aabb_max.x, tri.v2.x);
			node.aabb_max.y = std::max(node.aabb_max.y, tri.v2.y);
			node.aabb_max.z = std::max(node.aabb_max.z, tri.v2.z);
		}
	}

	void SAHBuilder::Subdivide(BVH& bvh, Triangle const* triangles, Uint32 node_idx)
	{
		BVHNode& node = bvh.nodes[node_idx];
		if (node.tri_count <= 2)
		{
			return;
		}

		// Find best split using binned SAH
		Int best_axis = -1;
		Float best_pos = 0, best_cost = 1e30f;

		for (Int axis = 0; axis < 3; axis++)
		{
			// Compute centroid bounds for this axis
			Float cmin = 1e30f, cmax = -1e30f;
			for (Uint32 i = 0; i < node.tri_count; i++)
			{
				Triangle const& tri = triangles[bvh.tri_indices[node.left_first + i]];
				Float c;
				if (axis == 0)      c = tri.centroid.x;
				else if (axis == 1) c = tri.centroid.y;
				else                c = tri.centroid.z;
				cmin = std::min(cmin, c);
				cmax = std::max(cmax, c);
			}

			if (cmin == cmax) continue;

			// Initialize bins
			struct Bin { AABB bounds; Uint32 count = 0; };
			Bin bins[NUM_BINS];

			Float scale = NUM_BINS / (cmax - cmin);

			// Populate bins
			for (Uint32 i = 0; i < node.tri_count; i++)
			{
				Triangle const& tri = triangles[bvh.tri_indices[node.left_first + i]];
				Float c;
				if (axis == 0)      c = tri.centroid.x;
				else if (axis == 1) c = tri.centroid.y;
				else                c = tri.centroid.z;

				Int bin_idx = std::min(static_cast<Int>((c - cmin) * scale), NUM_BINS - 1);
				bins[bin_idx].count++;
				bins[bin_idx].bounds.Grow(tri.v0);
				bins[bin_idx].bounds.Grow(tri.v1);
				bins[bin_idx].bounds.Grow(tri.v2);
			}

			// Gather data for the planes between bins using prefix/suffix sweep
			Float left_area[NUM_BINS - 1], right_area[NUM_BINS - 1];
			Uint32 left_count[NUM_BINS - 1], right_count[NUM_BINS - 1];
			AABB left_box, right_box;
			Uint32 left_sum = 0, right_sum = 0;
			for (Int i = 0; i < NUM_BINS - 1; i++)
			{
				left_sum += bins[i].count;
				left_count[i] = left_sum;
				left_box.Grow(bins[i].bounds);
				left_area[i] = left_box.Area();

				right_sum += bins[NUM_BINS - 1 - i].count;
				right_count[NUM_BINS - 2 - i] = right_sum;
				right_box.Grow(bins[NUM_BINS - 1 - i].bounds);
				right_area[NUM_BINS - 2 - i] = right_box.Area();
			}

			// Evaluate SAH cost for each plane
			Float plane_scale = (cmax - cmin) / NUM_BINS;
			for (Int i = 0; i < NUM_BINS - 1; i++)
			{
				Float cost = left_count[i] * left_area[i] + right_count[i] * right_area[i];
				if (cost < best_cost)
				{
					best_cost = cost;
					best_axis = axis;
					best_pos = cmin + plane_scale * (i + 1);
				}
			}
		}

		// SAH termination: don't split if it's not worthwhile
		Vector3 extent = node.aabb_max - node.aabb_min;
		Float parent_area = extent.x * extent.y + extent.y * extent.z + extent.z * extent.x;
		Float parent_cost = node.tri_count * parent_area;
		if (best_axis == -1 || best_cost >= parent_cost)
		{
			return;
		}

		Int axis = best_axis;
		Float split_pos = best_pos;

		// In-place partition
		Uint32 first_tri_idx = node.left_first;
		Int i = static_cast<Int>(first_tri_idx);
		Int j = i + static_cast<Int>(node.tri_count) - 1;
		while (i <= j)
		{
			Triangle const& tri = triangles[bvh.tri_indices[i]];
			Float centroid_val;
			if (axis == 0)      centroid_val = tri.centroid.x;
			else if (axis == 1) centroid_val = tri.centroid.y;
			else                centroid_val = tri.centroid.z;

			if (centroid_val < split_pos)
			{
				i++;
			}
			else
			{
				std::swap(bvh.tri_indices[i], bvh.tri_indices[j]);
				j--;
			}
		}

		Uint32 left_count = static_cast<Uint32>(i) - first_tri_idx;
		if (left_count == 0 || left_count == node.tri_count)
		{
			return;
		}

		Uint32 left_child_idx = bvh.nodes_used++;
		Uint32 right_child_idx = bvh.nodes_used++;

		bvh.nodes[left_child_idx].left_first = first_tri_idx;
		bvh.nodes[left_child_idx].tri_count = left_count;
		bvh.nodes[right_child_idx].left_first = static_cast<Uint32>(i);
		bvh.nodes[right_child_idx].tri_count = node.tri_count - left_count;

		node.left_first = left_child_idx;
		node.tri_count = 0;

		UpdateNodeBounds(bvh, triangles, left_child_idx);
		UpdateNodeBounds(bvh, triangles, right_child_idx);

		Subdivide(bvh, triangles, left_child_idx);
		Subdivide(bvh, triangles, right_child_idx);
	}
}
