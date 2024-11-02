#pragma once
#include <vector>

namespace amber
{
	struct Geometry
	{
		std::vector<Vector3>  vertices;
		std::vector<Vector3>  normals;
		std::vector<Vector2>  uvs;
		std::vector<Vector3u> indices;
	};

	struct Mesh
	{
		std::vector<Geometry> geometries;
		std::vector<Uint32>   material_ids;
	};

	struct Instance
	{
		Matrix transform;
		Uint64 mesh_id;
	};
}