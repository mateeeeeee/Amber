#pragma once
#include "TopDownBuilder.h"

namespace amber
{
	struct BinnedSAHPolicy
	{
		static constexpr Int NUM_BINS = 8;

		static std::optional<SplitResult> FindSplit(BVH const& bvh, Triangle const* triangles, BVHNode const& node)
		{
			Int   best_axis = -1;
			Float best_pos  = 0.0f;
			Float best_cost = BVH_INFINITY;

			for (Int axis = 0; axis < 3; axis++)
			{
				Float cmin = BVH_INFINITY, cmax = -BVH_INFINITY;
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

				struct Bin { AABB bounds; Uint32 count = 0; };
				Bin bins[NUM_BINS];

				Float scale = NUM_BINS / (cmax - cmin);
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

				Float  left_area[NUM_BINS - 1],  right_area[NUM_BINS - 1];
				Uint32 left_count[NUM_BINS - 1], right_count[NUM_BINS - 1];
				AABB   left_box, right_box;
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

				Float plane_scale = (cmax - cmin) / NUM_BINS;
				for (Int i = 0; i < NUM_BINS - 1; i++)
				{
					Float cost = left_count[i] * left_area[i] + right_count[i] * right_area[i];
					if (cost < best_cost)
					{
						best_cost = cost;
						best_axis = axis;
						best_pos  = cmin + plane_scale * (i + 1);
					}
				}
			}

			Vector3 extent      = node.aabb_max - node.aabb_min;
			Float   parent_area = extent.x * extent.y + extent.y * extent.z + extent.z * extent.x;
			Float   parent_cost = node.tri_count * parent_area;
			if (best_axis == -1 || best_cost >= parent_cost)
			{
				return std::nullopt;
			}

			return SplitResult{ best_axis, best_pos };
		}
	};

	using BinnedSAHBuilder = TopDownBuilder<BinnedSAHPolicy>;

	struct SweepSAHPolicy
	{
		static std::optional<SplitResult> FindSplit(BVH const& bvh, Triangle const* triangles, BVHNode const& node)
		{
			Int   best_axis = -1;
			Float best_pos  = 0.0f;
			Float best_cost = BVH_INFINITY;

			Uint32 count = node.tri_count;

			std::vector<Uint32> sorted(count);
			std::vector<AABB>   right_bounds(count);

			for (Int axis = 0; axis < 3; axis++)
			{
				for (Uint32 i = 0; i < count; i++) sorted[i] = bvh.tri_indices[node.left_first + i];
				std::sort(sorted.begin(), sorted.end(), [&](Uint32 a, Uint32 b)
				{
					Float ca, cb;
					if (axis == 0)      { ca = triangles[a].centroid.x; cb = triangles[b].centroid.x; }
					else if (axis == 1) { ca = triangles[a].centroid.y; cb = triangles[b].centroid.y; }
					else                { ca = triangles[a].centroid.z; cb = triangles[b].centroid.z; }
					return ca < cb;
				});

				AABB right_box;
				for (Int i = static_cast<Int>(count) - 1; i >= 0; i--)
				{
					Triangle const& tri = triangles[sorted[i]];
					right_box.Grow(tri.v0);
					right_box.Grow(tri.v1);
					right_box.Grow(tri.v2);
					right_bounds[i] = right_box;
				}

				AABB left_box;
				for (Uint32 i = 0; i < count - 1; i++)
				{
					Triangle const& tri = triangles[sorted[i]];
					left_box.Grow(tri.v0);
					left_box.Grow(tri.v1);
					left_box.Grow(tri.v2);

					Float cost = (i + 1) * left_box.Area() + (count - i - 1) * right_bounds[i + 1].Area();
					if (cost < best_cost)
					{
						best_cost = cost;
						best_axis = axis;
						Float ca, cb;
						if (axis == 0)      { ca = triangles[sorted[i]].centroid.x;     cb = triangles[sorted[i + 1]].centroid.x; }
						else if (axis == 1) { ca = triangles[sorted[i]].centroid.y;     cb = triangles[sorted[i + 1]].centroid.y; }
						else                { ca = triangles[sorted[i]].centroid.z;     cb = triangles[sorted[i + 1]].centroid.z; }
						best_pos = (ca + cb) * 0.5f;
					}
				}
			}

			Vector3 extent      = node.aabb_max - node.aabb_min;
			Float   parent_area = extent.x * extent.y + extent.y * extent.z + extent.z * extent.x;
			Float   parent_cost = node.tri_count * parent_area;
			if (best_axis == -1 || best_cost >= parent_cost)
			{
				return std::nullopt;
			}

			return SplitResult{ best_axis, best_pos };
		}
	};

	using SweepSAHBuilder = TopDownBuilder<SweepSAHPolicy>;
}
