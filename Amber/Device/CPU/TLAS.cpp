#include "TLAS.h"
#include <algorithm>
#include <numeric>

namespace amber
{
	static void BuildRecursive(TLAS& tlas, Uint32 node_idx, Uint32* blas_indices, Uint32 count)
	{
		TLASNode& node = tlas.nodes[node_idx];

		node.aabb_min = Vector3(BVH_INFINITY, BVH_INFINITY, BVH_INFINITY);
		node.aabb_max = Vector3(-BVH_INFINITY, -BVH_INFINITY, -BVH_INFINITY);
		for (Uint32 i = 0; i < count; i++)
		{
			AABB const& b = tlas.blas_list[blas_indices[i]].world_bounds;
			node.aabb_min.x = std::min(node.aabb_min.x, b.min.x);
			node.aabb_min.y = std::min(node.aabb_min.y, b.min.y);
			node.aabb_min.z = std::min(node.aabb_min.z, b.min.z);
			node.aabb_max.x = std::max(node.aabb_max.x, b.max.x);
			node.aabb_max.y = std::max(node.aabb_max.y, b.max.y);
			node.aabb_max.z = std::max(node.aabb_max.z, b.max.z);
		}

		if (count == 1)
		{
			node.left_first = blas_indices[0];
			node.is_leaf = 1;
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

		std::sort(blas_indices, blas_indices + count, [&](Uint32 a, Uint32 b)
		{
			AABB const& ba = tlas.blas_list[a].world_bounds;
			AABB const& bb = tlas.blas_list[b].world_bounds;
			Float ca = (axis == 0) ? (ba.min.x + ba.max.x) : (axis == 1) ? (ba.min.y + ba.max.y) : (ba.min.z + ba.max.z);
			Float cb = (axis == 0) ? (bb.min.x + bb.max.x) : (axis == 1) ? (bb.min.y + bb.max.y) : (bb.min.z + bb.max.z);
			return ca < cb;
		});

		Uint32 mid = count / 2;
		Uint32 left_idx = tlas.nodes_used++;
		Uint32 right_idx = tlas.nodes_used++;

		node.left_first = left_idx;
		node.is_leaf = 0;

		BuildRecursive(tlas, left_idx, blas_indices, mid);
		BuildRecursive(tlas, right_idx, blas_indices + mid, count - mid);
	}

	void Build(TLAS& tlas, BLAS* blas_list, Uint32 blas_count)
	{
		tlas.blas_list = blas_list;
		tlas.blas_count = blas_count;

		if (blas_count == 0)
		{
			return;
		}

		tlas.nodes.resize(2 * blas_count + 2);
		tlas.nodes_used = 1;

		std::vector<Uint32> indices(blas_count);
		std::iota(indices.begin(), indices.end(), 0);

		BuildRecursive(tlas, 0, indices.data(), blas_count);
	}

	Bool Intersect(TLAS const& tlas, Ray& ray, HitInfo& hit)
	{
		if (tlas.blas_count == 0)
		{
			return false;
		}

		TLASNode const* node = &tlas.nodes[0];
		TLASNode const* stack[64];
		Uint32 stack_ptr = 0;
		Bool found = false;

		while (true)
		{
			if (node->is_leaf)
			{
				if (amber::Intersect(tlas.blas_list[node->left_first], node->left_first, ray, hit))
				{
					found = true;
				}
				if (stack_ptr == 0)
				{
					break;
				}
				node = stack[--stack_ptr];
				continue;
			}

			TLASNode const* child1 = &tlas.nodes[node->left_first];
			TLASNode const* child2 = &tlas.nodes[node->left_first + 1];
			Float dist1 = IntersectAABB(ray, child1->aabb_min, child1->aabb_max);
			Float dist2 = IntersectAABB(ray, child2->aabb_min, child2->aabb_max);

			if (dist1 > dist2)
			{
				std::swap(dist1, dist2);
				std::swap(child1, child2);
			}

			if (dist1 == BVH_INFINITY)
			{
				if (stack_ptr == 0)
				{
					break;
				}
				node = stack[--stack_ptr];
			}
			else
			{
				node = child1;
				if (dist2 != BVH_INFINITY)
				{
					stack[stack_ptr++] = child2;
				}
			}
		}

		return found;
	}
}
