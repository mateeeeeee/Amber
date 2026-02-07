#include "AccelerationStructure.h"
#include "BVH/SAHBuilder.h"
#include "BVH/MedianSplitBuilder.h"

namespace amber
{
	void BuildBLAS(BLAS& blas, BLASBuildInput const& input)
	{
		blas.triangles.clear();
		for (Uint32 g = 0; g < input.geometry_count; g++)
		{
			GeometryDesc const& geom = input.geometries[g];
			for (Uint32 i = 0; i < geom.index_count; i++)
			{
				Vector3u const& idx = geom.indices[i];
				Triangle& tri = blas.triangles.emplace_back();
				tri.v0 = geom.vertices[idx.x];
				tri.v1 = geom.vertices[idx.y];
				tri.v2 = geom.vertices[idx.z];
				tri.centroid = (tri.v0 + tri.v1 + tri.v2) * (1.0f / 3.0f);
			}
		}

		if (HasAnyFlag(input.flags, BuildFlags::PreferFastBuild))
		{
			MedianSplitBuilder builder;
			builder.Build(blas.bvh, blas.triangles);
		}
		else
		{
			BinnedSAHBuilder builder;
			builder.Build(blas.bvh, blas.triangles);
		}
	}

	void BuildTLAS(TLAS& tlas, BLAS* blas_list, TLASBuildInput const& input)
	{
		Build(tlas, blas_list, input.instance_count);
	}
}
