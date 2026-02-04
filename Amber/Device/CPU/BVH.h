#pragma once
#include <vector>
#include "Intersection.h"

namespace amber
{
	struct BVHNode
	{
		Vector3 aabb_min;
		Vector3 aabb_max;
		Uint32 left_first;
		Uint32 tri_count;

		Bool IsLeaf() const { return tri_count > 0; }
	};

	struct BVHBuildData
	{
		std::vector<BVHNode> nodes;
		std::vector<Uint32> tri_indices;
		Uint32 nodes_used = 0;
	};

	class BVH
	{
	public:
		BVH() = default;

		template<typename BuildPolicyT>
		void Build(std::vector<Triangle> const& triangles)
		{
			this->triangles = &triangles;
			BuildPolicyT::Build(data, triangles);
		}

		Bool Intersect(Ray const& ray, HitInfo& hit) const;

	private:
		BVHBuildData data;
		std::vector<Triangle> const* triangles = nullptr;

	private:
		void IntersectRecursive(Ray const& ray, Uint32 node_idx, HitInfo& hit, Bool& found) const;
	};
}
