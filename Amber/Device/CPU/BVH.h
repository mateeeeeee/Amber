#pragma once
#include <vector>
#include "Intersection.h"

namespace amber
{
	struct BVHNode
	{
		Vector3 aabb_min;
		Vector3 aabb_max;
		Uint32 left_first;  // If leaf (tri_count > 0): first triangle index. Otherwise: left child index
		Uint32 tri_count;   // If 0: interior node. Otherwise: leaf with tri_count triangles

		Bool IsLeaf() const { return tri_count > 0; }
	};

	class BVH
	{
	public:
		BVH() = default;

		void Build(std::vector<Triangle> const& triangles);
		Bool Intersect(Ray const& ray, HitInfo& hit) const;

	private:
		void UpdateNodeBounds(Uint32 node_idx);
		void Subdivide(Uint32 node_idx);
		void IntersectRecursive(Ray const& ray, Uint32 node_idx, HitInfo& hit, Bool& found) const;

		std::vector<BVHNode> nodes;
		std::vector<Triangle> const* triangles = nullptr;
		std::vector<Uint32> tri_indices;
		Uint32 nodes_used = 0;
	};
}
