#include "AccelerationStructure.h"
#include "BVH/SAHBuilder.h"
#include "BVH/MedianSplitBuilder.h"
#include "BVH/PLOCBuilder.h"

namespace amber
{
	void BuildBLAS(BLAS& blas, BLASBuildInput const& input)
	{
		blas.triangles.clear();
		blas.face_indices.clear();
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
				blas.face_indices.push_back(i);
			}
		}

		if (HasAnyFlag(input.flags, BuildFlags::PreferFastBuild))
		{
			MedianSplitBuilder builder;
			builder.Build(blas.bvh, std::span(blas.triangles));
		}
		else
		{
			PLOCBuilder builder;
			builder.Build(blas.bvh, std::span(blas.triangles));
		}
	}

	void BuildTLAS(TLAS& tlas, BLAS* blas_list, Uint32 blas_count, TLASBuildInput const& input)
	{
		if (input.instance_count == 0)
		{
			return;
		}

		tlas.instances.resize(input.instance_count);
		for (Uint32 i = 0; i < input.instance_count; i++)
		{
			InstanceDesc const& desc = input.instances[i];
			BLAS const& blas = blas_list[desc.blas_index];

			BLASInstance& inst    = tlas.instances[i];
			inst.blas         = &blas;
			inst.inv_transform = desc.transform.Inverse();
			inst.instance_id  = desc.instance_id;
			inst.mask         = desc.mask;
			inst.flags        = desc.flags;

			Vector3 bmin = blas.bvh.nodes[0].aabb_min;
			Vector3 bmax = blas.bvh.nodes[0].aabb_max;
			inst.world_bounds = AABB();
			for (Int j = 0; j < 8; j++)
			{
				Vector3 corner(
					(j & 1) ? bmax.x : bmin.x,
					(j & 2) ? bmax.y : bmin.y,
					(j & 4) ? bmax.z : bmin.z
				);
				inst.world_bounds.Grow(Vector3::Transform(corner, desc.transform));
			}
		}

		if (HasAnyFlag(input.flags, BuildFlags::PreferFastBuild))
		{
			MedianSplitBuilder builder;
			builder.Build(tlas.bvh, std::span(tlas.instances));
		}
		else
		{
			PLOCBuilder builder;
			builder.Build(tlas.bvh, std::span(tlas.instances));
		}
	}
}
