#pragma once
#include "BLAS.h"
#include "TLAS.h"
#include "RayFlags.h"

namespace amber
{
	enum class BuildFlags : Uint8
	{
		None            = 0,
		PreferFastTrace = 1 << 0,
		PreferFastBuild = 1 << 1,
		AllowUpdate     = 1 << 2,
	};
	ENABLE_ENUM_BIT_OPERATORS(BuildFlags);

	struct GeometryDesc
	{
		Vector3 const*   vertices     = nullptr;
		Uint32           vertex_count = 0;
		Vector3u const*  indices      = nullptr;
		Uint32           index_count  = 0;   
		RayGeometryFlags flags        = RayGeometryFlags::Opaque;
	};

	struct BLASBuildInput
	{
		GeometryDesc const* geometries     = nullptr;
		Uint32              geometry_count = 0;
		BuildFlags          flags          = BuildFlags::PreferFastTrace;
	};

	struct InstanceDesc
	{
		Matrix           transform   = Matrix::Identity;
		Uint32           blas_index  = 0;
		Uint32           instance_id = 0;
		Uint8            mask        = 0xFF;
		RayGeometryFlags flags       = RayGeometryFlags::Opaque;
	};

	struct TLASBuildInput
	{
		InstanceDesc const* instances      = nullptr;
		Uint32              instance_count = 0;
		BuildFlags          flags          = BuildFlags::PreferFastTrace;
	};

	void BuildBLAS(BLAS& blas, BLASBuildInput const& input);
	void BuildTLAS(TLAS& tlas, BLAS* blas_list, Uint32 blas_count, TLASBuildInput const& input);
}
