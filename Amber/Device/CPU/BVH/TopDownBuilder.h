#pragma once
#include <optional>
#include <span>
#include "BVH.h"
#include "SpatialTraits.h"

namespace amber
{
	struct SplitResult
	{
		Int   axis;
		Float pos;
	};

	template<typename NodeT, typename SplitPolicyT>
	concept TopDownSplitPolicy = requires(BVH const& bvh, NodeT const* nodes, BVHNode const& node)
	{
		{ SplitPolicyT::FindSplit(bvh, nodes, node) } -> std::same_as<std::optional<SplitResult>>;
	};

	template<typename NodeT, typename SplitPolicyT> requires TopDownSplitPolicy<NodeT, SplitPolicyT>
	class TopDownBuilder
	{
		using Traits = SpatialTraits<NodeT>;
	public:
		void Build(BVH& bvh, std::span<NodeT const> nodes);

	private:
		void UpdateNodeBounds(BVH& bvh, NodeT const* nodes, Uint32 node_idx);
		void Subdivide(BVH& bvh, NodeT const* nodes, Uint32 node_idx);
	};
}
