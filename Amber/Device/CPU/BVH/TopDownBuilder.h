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
	concept TopDownSplitPolicy = requires(BVH2 const& bvh, NodeT const* nodes, BVH2Node const& node)
	{
		{ SplitPolicyT::FindSplit(bvh, nodes, node) } -> std::same_as<std::optional<SplitResult>>;
	};

	template<typename SplitPolicyT>
	class TopDownBuilder
	{
	public:
		TopDownBuilder();

		template<typename NodeT> requires TopDownSplitPolicy<NodeT, SplitPolicyT>
		void Build(BVH2& bvh, std::span<NodeT> nodes);

	private:
		Uint32 max_leaf_prims;

		template<typename NodeT>
		void UpdateNodeBounds(BVH2& bvh, NodeT const* nodes, Uint32 node_idx);

		template<typename NodeT>
		void Subdivide(BVH2& bvh, NodeT const* nodes, Uint32 node_idx);
	};

	struct BinnedSAHPolicy;
	struct SweepSAHPolicy;
	struct MedianSplitPolicy;
	extern template class TopDownBuilder<BinnedSAHPolicy>;
	extern template class TopDownBuilder<SweepSAHPolicy>;
	extern template class TopDownBuilder<MedianSplitPolicy>;
}
