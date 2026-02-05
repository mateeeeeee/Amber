#pragma once
#include <vector>
#include "Device/BVH/BVHData.h"
#include "Device/BVH/Intersection.h"

namespace amber
{
	struct CpuBVHBuildData
	{
		std::vector<BVHNode> nodes;
		std::vector<Uint32> tri_indices;
		Uint32 nodes_used = 0;
	};

	class CpuBVH
	{
	public:
		CpuBVH() = default;

		template<template<typename> typename BuildPolicyT>
		void Build(std::vector<Triangle> const& triangles)
		{
			this->triangles = &triangles;
			BuildPolicyT<CpuBVHBuildData>::Build(data, triangles.data(), static_cast<Uint32>(triangles.size()));
		}

		Bool Intersect(Ray const& ray, HitInfo& hit) const;
		CpuBVHBuildData const& GetData() const { return data; }

	private:
		CpuBVHBuildData data;
		std::vector<Triangle> const* triangles = nullptr;

	private:
		void IntersectRecursive(Ray const& ray, Uint32 node_idx, HitInfo& hit, Bool& found) const;
	};
}
