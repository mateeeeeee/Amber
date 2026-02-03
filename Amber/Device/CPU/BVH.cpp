#include "BVH.h"
#include <algorithm>

// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/

namespace amber
{
	void BVH::Build(std::vector<Triangle> const& tris)
	{
		triangles = &tris;
		Uint32 n = static_cast<Uint32>(tris.size());
		if (n == 0)
		{
			return;
		}

		tri_indices.resize(n);
		for (Uint32 i = 0; i < n; i++)
		{
			tri_indices[i] = i;
		}

		nodes.resize(n * 2 - 1);
		nodes_used = 1;

		BVHNode& root = nodes[0];
		root.left_first = 0;
		root.tri_count = n;
		UpdateNodeBounds(0);
		Subdivide(0);
	}

	void BVH::UpdateNodeBounds(Uint32 node_idx)
	{
		BVHNode& node = nodes[node_idx];
		node.aabb_min = Vector3(1e30f, 1e30f, 1e30f);
		node.aabb_max = Vector3(-1e30f, -1e30f, -1e30f);
		for (Uint32 i = 0; i < node.tri_count; i++)
		{
			Uint32 tri_idx = tri_indices[node.left_first + i];
			Triangle const& tri = (*triangles)[tri_idx];

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

	void BVH::Subdivide(Uint32 node_idx)
	{
		BVHNode& node = nodes[node_idx];
		if (node.tri_count <= 2)
		{
			return;
		}

		// Determine split axis (longest extent)
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
			Triangle const& tri = (*triangles)[tri_indices[i]];
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
				std::swap(tri_indices[i], tri_indices[j]);
				j--;
			}
		}

		Uint32 left_count = static_cast<Uint32>(i) - first_tri_idx;
		if (left_count == 0 || left_count == node.tri_count)
		{
			return;
		}

		Uint32 left_child_idx = nodes_used++;
		Uint32 right_child_idx = nodes_used++;

		nodes[left_child_idx].left_first = first_tri_idx;
		nodes[left_child_idx].tri_count = left_count;
		nodes[right_child_idx].left_first = static_cast<Uint32>(i);
		nodes[right_child_idx].tri_count = node.tri_count - left_count;

		node.left_first = left_child_idx;
		node.tri_count = 0;

		UpdateNodeBounds(left_child_idx);
		UpdateNodeBounds(right_child_idx);

		Subdivide(left_child_idx);
		Subdivide(right_child_idx);
	}

	Bool BVH::Intersect(Ray const& ray, HitInfo& hit) const
	{
		Bool found = false;
		IntersectRecursive(ray, 0, hit, found);
		return found;
	}

	void BVH::IntersectRecursive(Ray const& ray, Uint32 node_idx, HitInfo& hit, Bool& found) const
	{
		BVHNode const& node = nodes[node_idx];
		if (!IntersectAABB(ray, node.aabb_min, node.aabb_max, hit.t))
		{
			return;
		}

		if (node.IsLeaf())
		{
			for (Uint32 i = 0; i < node.tri_count; i++)
			{
				Uint32 tri_idx = tri_indices[node.left_first + i];
				Triangle const& tri = (*triangles)[tri_idx];
				HitInfo temp_hit;
				if (IntersectTriangle(ray, tri, temp_hit) && temp_hit.t < hit.t)
				{
					hit = temp_hit;
					hit.tri_idx = tri_idx;
					found = true;
				}
			}
		}
		else
		{
			IntersectRecursive(ray, node.left_first, hit, found);
			IntersectRecursive(ray, node.left_first + 1, hit, found);
		}
	}
}
