#pragma once
#include <vector>
#include "Device/CPU/BVH/Intersection.h"

namespace amber
{
	template<Uint N>
	struct BVHNode
	{
		static_assert(N == 2 || N == 4 || N == 8);

		Vector3 aabb_min;
		Vector3 aabb_max;
		Uint32  child_count; 
		union 
		{
			Uint32 children[N];
			struct 
			{
				Uint32 first_prim;
				Uint32 prim_count;
			};
		};

		Bool IsLeaf() const { return child_count == 0; }
	};

	template<Uint N>
	struct BVH
	{
		static_assert(N == 2 || N == 4 || N == 8);

		std::vector<BVHNode<N>> nodes;
		std::vector<Uint32>     prim_indices;
		Uint32                  nodes_used = 0;
	};

	using BVH2     = BVH<2>;
	using BVH4     = BVH<4>;
	using BVH8     = BVH<8>;
	using BVH2Node = BVHNode<2>;
	using BVH4Node = BVHNode<4>;
	using BVH8Node = BVHNode<8>;
}
