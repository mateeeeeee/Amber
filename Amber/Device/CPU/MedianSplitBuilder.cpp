#include "MedianSplitBuilder.h"
#include <algorithm>

// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/

namespace amber
{
	void MedianSplitBuilder::Build(BVH& bvh, std::vector<Triangle> const& triangles)
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

	void MedianSplitBuilder::UpdateNodeBounds(BVH& bvh, Triangle const* triangles, Uint32 node_idx)
	{
		BVHNode& node = bvh.nodes[node_idx];
		node.aabb_min = Vector3(BVH_INFINITY, BVH_INFINITY, BVH_INFINITY);
		node.aabb_max = Vector3(-BVH_INFINITY, -BVH_INFINITY, -BVH_INFINITY);
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

	void MedianSplitBuilder::Subdivide(BVH& bvh, Triangle const* triangles, Uint32 node_idx)
	{
		BVHNode& node = bvh.nodes[node_idx];
		if (node.tri_count <= 2)
		{
			return;
		}

		Vector3 extent = node.aabb_max - node.aabb_min;
		Int axis = 0;
		if (extent.y > extent.x)
		{
			axis = 1;
		}
		if (extent.z > (axis == 0 ? extent.x : extent.y))
		{
			axis = 2;
		}

		Float split_pos;
		if (axis == 0)
		{
			split_pos = node.aabb_min.x + extent.x * 0.5f;
		}
		else if (axis == 1)
		{
			split_pos = node.aabb_min.y + extent.y * 0.5f;
		}
		else
		{
			split_pos = node.aabb_min.z + extent.z * 0.5f;
		}

		// In-place partition
		Uint32 first_tri_idx = node.left_first;
		Int i = static_cast<Int>(first_tri_idx);
		Int j = i + static_cast<Int>(node.tri_count) - 1;
		while (i <= j)
		{
			Triangle const& tri = triangles[bvh.tri_indices[i]];
			Float centroid_val;
			if (axis == 0)
			{
				centroid_val = tri.centroid.x;
			}
			else if (axis == 1)
			{
				centroid_val = tri.centroid.y;
			}
			else
			{
				centroid_val = tri.centroid.z;
			}

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
