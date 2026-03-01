#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include "Core/Log.h"
#include "Core/Paths.h"
#include "Utilities/CLIParser.h"
#include "Utilities/Timer.h"
#include "Utilities/Random.h"
#include "Utilities/JsonUtil.h"
#include "Scene/Scene.h"
#include "Scene/Mesh.h"
#include "Device/CPU/BVH/BVH.h"
#include "Device/CPU/BVH/Primitives.h"
#include "Device/CPU/BVH/Intersection.h"
#include "Device/CPU/BVH/TopDownBuilder.h"
#include "Device/CPU/BVH/SAHBuilder.h"
#include "Device/CPU/BVH/MedianSplitBuilder.h"
#include "Device/CPU/BVH/PLOCBuilder.h"
#include "Device/CPU/BVH/Collapse.h"
#include "Device/CPU/BVH/Traversal.h"
#include "Device/CPU/BVH/Stats.h"

using namespace amber;

static constexpr Float PI = 3.14159265358979323846f;
static constexpr Uint32 SEED = 42;

struct BenchmarkResult
{
	std::string builder_name;
	Float build_ms;
	Float collapse_ms;
	Float sah_cost;
	Uint32 max_depth;
	Float trace_ms;
	Float avg_steps;
	Float avg_prim_tests;
};

static std::vector<Triangle> ExtractTriangles(Scene const& scene)
{
	// Match CpuPathTracer::BuildAccelerationStructures
	std::vector<Geometry const*> flat_geoms;
	for (Mesh const& mesh : scene.meshes)
	{
		for (Geometry const& geom : mesh.geometries)
		{
			flat_geoms.push_back(&geom);
		}
	}

	std::vector<Triangle> triangles;
	for (Instance const& instance : scene.instances)
	{
		Geometry const& geom = *flat_geoms[instance.mesh_id];
		for (Vector3u const& idx : geom.indices)
		{
			Triangle tri;
			tri.v0 = Vector3::Transform(geom.vertices[idx.x], instance.transform);
			tri.v1 = Vector3::Transform(geom.vertices[idx.y], instance.transform);
			tri.v2 = Vector3::Transform(geom.vertices[idx.z], instance.transform);
			tri.centroid = (tri.v0 + tri.v1 + tri.v2) * (1.0f / 3.0f);
			triangles.push_back(tri);
		}
	}
	return triangles;
}

static AABB ComputeSceneAABB(std::vector<Triangle> const& triangles)
{
	AABB aabb;
	for (Triangle const& tri : triangles)
	{
		aabb.Grow(tri.v0);
		aabb.Grow(tri.v1);
		aabb.Grow(tri.v2);
	}
	return aabb;
}

static std::vector<Ray> GenerateRays(AABB const& aabb, Uint32 width, Uint32 height)
{
	RealRandomGenerator<Float> jitter(0.0f, 1.0f, std::mt19937(SEED));

	Vector3 center;
	center.x = (aabb.min.x + aabb.max.x) * 0.5f;
	center.y = (aabb.min.y + aabb.max.y) * 0.5f;
	center.z = (aabb.min.z + aabb.max.z) * 0.5f;

	Vector3 extent;
	extent.x = aabb.max.x - aabb.min.x;
	extent.y = aabb.max.y - aabb.min.y;
	extent.z = aabb.max.z - aabb.min.z;

	Float radius = std::sqrt(extent.x * extent.x + extent.y * extent.y + extent.z * extent.z) * 0.5f;

	// Camera on the enclosing sphere, looking at center
	Vector3 cam_pos;
	cam_pos.x = center.x + radius;
	cam_pos.y = center.y + radius * 0.5f;
	cam_pos.z = center.z + radius;

	Vector3 forward;
	forward.x = center.x - cam_pos.x;
	forward.y = center.y - cam_pos.y;
	forward.z = center.z - cam_pos.z;
	Float fwd_len = std::sqrt(forward.x * forward.x + forward.y * forward.y + forward.z * forward.z);
	forward.x /= fwd_len;
	forward.y /= fwd_len;
	forward.z /= fwd_len;

	Vector3 world_up(0.0f, 1.0f, 0.0f);
	Vector3 right = Vector3::Cross(forward, world_up);
	Float right_len = std::sqrt(right.x * right.x + right.y * right.y + right.z * right.z);
	right.x /= right_len;
	right.y /= right_len;
	right.z /= right_len;

	Vector3 up = Vector3::Cross(right, forward);

	Float fov = PI / 4.0f; // 45 degrees
	Float aspect = (Float)width / (Float)height;
	Float half_h = std::tan(fov * 0.5f);
	Float half_w = half_h * aspect;

	std::vector<Ray> rays;
	rays.reserve(width * height);
	for (Uint32 y = 0; y < height; y++)
	{
		for (Uint32 x = 0; x < width; x++)
		{
			Float u = (2.0f * ((Float)x + jitter()) / (Float)width - 1.0f) * half_w;
			Float v = (2.0f * ((Float)y + jitter()) / (Float)height - 1.0f) * half_h;

			Vector3 dir;
			dir.x = forward.x + u * right.x + v * up.x;
			dir.y = forward.y + u * right.y + v * up.y;
			dir.z = forward.z + u * right.z + v * up.z;
			Float len = std::sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
			dir.x /= len;
			dir.y /= len;
			dir.z /= len;

			rays.emplace_back(cam_pos, dir);
		}
	}
	return rays;
}

template<typename BuilderT>
static BenchmarkResult RunBenchmark(
	std::string const& name,
	std::vector<Triangle> triangles,
	std::vector<Ray> const& rays)
{
	BenchmarkResult result{};
	result.builder_name = name;

	BVH2 bvh2;
	{
		Timer timer;
		BuilderT builder;
		builder.Build(bvh2, std::span(triangles));
		result.build_ms = timer.Peek() / 1000.0f;
	}

	BVH8 bvh8;
	{
		Timer timer;
		Collapse(bvh2, bvh8);
		result.collapse_ms = timer.Peek() / 1000.0f;
	}

	BVHStats stats = ComputeStats(bvh8);
	result.sah_cost = stats.sah_cost;
	result.max_depth = stats.max_depth;
	{
		Timer timer;
		for (Ray const& src_ray : rays)
		{
			Ray ray = src_ray;
			HitInfo hit;
			Intersect(bvh8, triangles, ray, hit);
		}
		result.trace_ms = timer.Peek() / 1000.0f;
	}
	Uint64 total_steps = 0;
	Uint64 total_prim_tests = 0;
	for (Ray const& ray : rays)
	{
		total_steps += CountTraversalSteps(bvh8, ray);
		total_prim_tests += CountPrimTests(bvh8, ray);
	}
	result.avg_steps = (Float)total_steps / (Float)rays.size();
	result.avg_prim_tests = (Float)total_prim_tests / (Float)rays.size();
	return result;
}

static void PrintTable(std::vector<BenchmarkResult> const& results)
{
	std::printf("\n");
	std::printf("%-16s | %10s | %13s | %8s | %5s | %10s | %9s | %14s\n",
		"Builder", "Build (ms)", "Collapse (ms)", "SAH Cost", "Depth", "Trace (ms)", "Avg Steps", "Avg Prim Tests");
	std::printf("-----------------+------------+---------------+----------+-------+------------+-----------+---------------\n");
	for (auto const& r : results)
	{
		std::printf("%-16s | %10.2f | %13.2f | %8.2f | %5u | %10.2f | %9.2f | %14.2f\n",
			r.builder_name.c_str(),
			r.build_ms, r.collapse_ms, r.sah_cost, r.max_depth,
			r.trace_ms, r.avg_steps, r.avg_prim_tests);
	}
	std::printf("\n");
}

static void WriteCSV(std::string const& path, std::vector<BenchmarkResult> const& results)
{
	std::ofstream f(path);
	f << "Builder,Build (ms),Collapse (ms),SAH Cost,Depth,Trace (ms),Avg Steps,Avg Prim Tests\n";
	for (auto const& r : results)
	{
		f << r.builder_name << ","
		  << r.build_ms << "," << r.collapse_ms << ","
		  << r.sah_cost << "," << r.max_depth << ","
		  << r.trace_ms << "," << r.avg_steps << "," << r.avg_prim_tests << "\n";
	}
	std::printf("Results written to %s\n", path.c_str());
}

int main(Int argc, Char* argv[])
{
	g_Log.Initialize("bvhbenchmark.log", LogLevel::Info);

	CLIParser parser;
	parser.AddArg(true, "-s", "--scene");
	parser.AddArg(true, "-w", "--width");
	parser.AddArg(true, "-h", "--height");
	parser.AddArg(true, "-b", "--builder");
	parser.AddArg(true, "-o", "--output");
	parser.AddArg(false, "--help");

	CLIParseResult cli = parser.Parse(argc, argv);
	if (cli["--help"])
	{
		std::printf("Usage: BVHBenchmark --scene <scene.json> [options]\n");
		std::printf("  -s, --scene    Scene config file (relative to Saved/Scenes/)\n");
		std::printf("  -w, --width    Ray grid width  (default: 256)\n");
		std::printf("  -h, --height   Ray grid height (default: 256)\n");
		std::printf("  -b, --builder  Builder name or 'all' (default: all)\n");
		std::printf("                 Options: MedianSplit, BinnedSAH, SweepSAH, PLOC\n");
		std::printf("  -o, --output   Output CSV file path\n");
		std::printf("      --help     Show this help\n");
		return 0;
	}

	std::string scene_config = cli["--scene"].AsStringOr("sponza.json");
	Uint32 ray_width = cli["--width"].AsIntOr(256);
	Uint32 ray_height = cli["--height"].AsIntOr(256);
	std::string builder_name = cli["--builder"].AsStringOr("all");
	std::string output_file = cli["--output"].AsStringOr("");

	json json_scene;
	try
	{
		JsonParams scene_params = json::parse(std::ifstream(paths::SceneDir + scene_config));
		json_scene = scene_params.FindJson("scene");
	}
	catch (json::parse_error const& e)
	{
		std::fprintf(stderr, "JSON parsing error: %s\n", e.what());
		return 1;
	}

	JsonParams scene_params(json_scene);
	std::string model_file;
	if (!scene_params.Find<std::string>("model file", model_file))
	{
		std::fprintf(stderr, "No 'model file' in scene config\n");
		return 1;
	}

	Float model_scale = scene_params.FindOr<Float>("model scale", 1.0f);
	std::string full_model_path = paths::ModelDir + model_file;
	std::printf("Loading scene: %s (scale: %.1f)\n", model_file.c_str(), model_scale);

	std::unique_ptr<Scene> scene;
	try
	{
		scene = LoadScene(full_model_path.c_str(), "", model_scale);
	}
	catch (std::runtime_error const& e)
	{
		std::fprintf(stderr, "Failed to load scene: %s\n", e.what());
		return 1;
	}
	if (!scene)
	{
		std::fprintf(stderr, "Failed to load scene\n");
		return 1;
	}

	std::vector<Triangle> triangles = ExtractTriangles(*scene);
	Uint32 ray_count = ray_width * ray_height;
	std::printf("Scene: %s (%zu triangles)\n", model_file.c_str(), triangles.size());
	std::printf("Rays: %ux%u (%u total)\n", ray_width, ray_height, ray_count);

	AABB aabb = ComputeSceneAABB(triangles);
	std::vector<Ray> rays = GenerateRays(aabb, ray_width, ray_height);

	std::vector<BenchmarkResult> results;
	auto ShouldRun = [&](std::string const& name)
	{
		return builder_name == "all" || builder_name == name;
	};

	if (ShouldRun("MedianSplit"))
	{
		std::printf("Running MedianSplit...\n");
		results.push_back(RunBenchmark<MedianSplitBuilder>("MedianSplit", triangles, rays));
	}
	if (ShouldRun("BinnedSAH"))
	{
		std::printf("Running BinnedSAH...\n");
		results.push_back(RunBenchmark<BinnedSAHBuilder>("BinnedSAH", triangles, rays));
	}
	if (ShouldRun("SweepSAH"))
	{
		std::printf("Running SweepSAH...\n");
		results.push_back(RunBenchmark<SweepSAHBuilder>("SweepSAH", triangles, rays));
	}
	if (ShouldRun("PLOC"))
	{
		std::printf("Running PLOC...\n");
		results.push_back(RunBenchmark<PLOCBuilder>("PLOC", triangles, rays));
	}

	if (results.empty())
	{
		std::fprintf(stderr, "Unknown builder: %s\n", builder_name.c_str());
		return 1;
	}

	PrintTable(results);

	if (!output_file.empty())
	{
		WriteCSV(output_file, results);
	}

	g_Log.Destroy();
	return 0;
}
