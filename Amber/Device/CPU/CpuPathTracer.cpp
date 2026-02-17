#include "CpuPathTracer.h"
#include "PathTracingUtils.h"
#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include "Core/Log.h"
#include "Utilities/ImageUtil.h"
#include "Utilities/Timer.h"
#include <cmath>

namespace amber
{

	CpuPathTracer::CpuPathTracer(Uint32 width, Uint32 height, PathTracerConfig const& config, std::unique_ptr<Scene>&& _scene)
		: width(width), height(height), scene(std::move(_scene)), framebuffer(height, width)
	{
		g_ThreadPool.Initialize();
		framebuffer.Clear(RGBA8(0, 0, 0, 255));
		Timer<std::chrono::milliseconds> timer;
		BuildAccelerationStructures();
		AMBER_INFO_LOG("Built acceleration structures in %lldms", timer.Elapsed());
		AMBER_INFO_LOG("CPU PathTracer initialized with %u triangles, %zu unique BLASes", triangle_count, blas_list.size());
	}

	CpuPathTracer::~CpuPathTracer()
	{
		g_ThreadPool.Destroy();
	}

	void CpuPathTracer::BuildAccelerationStructures()
	{
		std::vector<BLAS> flat_blas;
		std::vector<Uint32> blas_by_geom_id;
		triangle_count = 0;

		for (Mesh const& mesh : scene->meshes)
		{
			for (Uint32 g = 0; g < static_cast<Uint32>(mesh.geometries.size()); g++)
			{
				Geometry const& geom = mesh.geometries[g];
				Uint32 blas_idx = static_cast<Uint32>(flat_blas.size());
				blas_by_geom_id.push_back(blas_idx);

				GeometryDesc geom_desc{};
				geom_desc.vertices    = geom.vertices.data();
				geom_desc.vertex_count = static_cast<Uint32>(geom.vertices.size());
				geom_desc.indices     = geom.indices.data();
				geom_desc.index_count = static_cast<Uint32>(geom.indices.size());

				BLASBuildInput build_input{};
				build_input.geometries     = &geom_desc;
				build_input.geometry_count = 1;
				build_input.flags          = BuildFlags::PreferFastTrace;

				BLAS blas{};
				AMBER_INFO_LOG("Building BLAS %u...", blas_idx);
				BuildBLAS(blas, build_input);
				triangle_count += static_cast<Uint32>(blas.triangles.size());
				flat_blas.push_back(std::move(blas));
				blas_geometries.push_back(&geom);
				blas_material_ids.push_back(mesh.material_ids.empty() ? 0 : mesh.material_ids[g]);
			}
		}

		std::vector<InstanceDesc> instance_descs;
		instance_descs.reserve(scene->instances.size());
		for (Uint32 i = 0; i < static_cast<Uint32>(scene->instances.size()); i++)
		{
			Instance const& scene_instance = scene->instances[i];
			InstanceDesc desc{};
			desc.transform   = scene_instance.transform;
			desc.blas_index  = blas_by_geom_id[scene_instance.mesh_id];
			desc.instance_id = i;
			instance_descs.push_back(desc);
		}

		blas_list = std::move(flat_blas);

		TLASBuildInput tlas_input{};
		tlas_input.instances      = instance_descs.data();
		tlas_input.instance_count = static_cast<Uint32>(instance_descs.size());
		tlas_input.flags          = BuildFlags::PreferFastTrace;

		AMBER_INFO_LOG("Building TLAS over %zu instances...", instance_descs.size());
		BuildTLAS(tlas, blas_list.data(), static_cast<Uint32>(blas_list.size()), tlas_input);

		textures.reserve(scene->textures.size());
		for (Image const& img : scene->textures)
		{
			Texture tex{};
			tex.data   = img.GetData();
			tex.width  = static_cast<Uint32>(img.GetWidth());
			tex.height = static_cast<Uint32>(img.GetHeight());
			tex.format = img.IsSRGB() ? TextureFormat::RGBA8_SRGB : TextureFormat::RGBA8;
			textures.push_back(tex);
		}

		if (scene->environment)
		{
			Image const& env = *scene->environment;
			env_texture.width  = static_cast<Uint32>(env.GetWidth());
			env_texture.height = static_cast<Uint32>(env.GetHeight());
			if (env.IsHDR())
			{
				env_texture.data   = env.GetData<Float>();
				env_texture.format = TextureFormat::RGBA32F;
			}
			else
			{
				env_texture.data   = env.GetData();
				env_texture.format = env.IsSRGB() ? TextureFormat::RGBA8_SRGB : TextureFormat::RGBA8;
			}
		}
	}

	void CpuPathTracer::Update(Float dt)
	{
	}

	void CpuPathTracer::Render(Camera const& camera)
	{
		if (camera.IsChanged())
		{
			frame_index = 0;
		}

		Timer<std::chrono::microseconds> timer;

		Vector3 origin = camera.GetPosition();
		Vector3 U, V, W;
		camera.GetFrame(U, V, W);

		Float fov_rad = camera.GetFovY() * (3.14159265359f / 180.0f);
		Float tan_half_fov = std::tan(fov_rad * 0.5f);
		Float aspect = camera.GetAspectRatio();

		Uint32 tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
		Uint32 tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;

		std::vector<std::future<void>> futures;
		futures.reserve(tiles_x * tiles_y);
		for (Uint32 ty = 0; ty < tiles_y; ++ty)
		{
			for (Uint32 tx = 0; tx < tiles_x; ++tx)
			{
				futures.push_back(g_ThreadPool.Submit([this, tx, ty, origin, U, V, W, tan_half_fov, aspect]()
				{
					Uint32 x_begin = tx * TILE_SIZE;
					Uint32 y_begin = ty * TILE_SIZE;
					Uint32 x_end = std::min(x_begin + TILE_SIZE, width);
					Uint32 y_end = std::min(y_begin + TILE_SIZE, height);

					for (Uint32 y = y_begin; y < y_end; ++y)
					{
						for (Uint32 x = x_begin; x < x_end; ++x)
						{
							Float ndc_x = (2.0f * (x + 0.5f) / width - 1.0f) * aspect * tan_half_fov;
							Float ndc_y = (1.0f - 2.0f * (y + 0.5f) / height) * tan_half_fov;

							Uint32 rng = PcgHash(x + PcgHash(y + PcgHash(frame_index)));

							Ray ray(origin, (W + U * ndc_x + V * ndc_y).Normalized());
							Vector3 throughput(1.0f, 1.0f, 1.0f);
							Vector3 radiance(0.0f, 0.0f, 0.0f);

							for (Uint32 depth = 0; depth < MAX_DEPTH; ++depth)
							{
								HitInfo hit;
								if (!Intersect(tlas, ray, hit))
								{
									radiance += throughput * SampleEnvironment(env_texture, ray.direction);
									break;
								}

								BLASInstance const& inst = tlas.instances[hit.instance_idx];
								Uint32 blas_idx = static_cast<Uint32>(inst.blas - blas_list.data());
								Geometry const& geom = *blas_geometries[blas_idx];

								Uint32 face = inst.blas->face_indices[hit.tri_idx];
								Vector3u const& vidx = geom.indices[face];
								Float bw = 1.0f - hit.u - hit.v;

								Vector3 normal;
								if (!geom.normals.empty())
								{
									Vector3 local_normal = (geom.normals[vidx.x] * bw + geom.normals[vidx.y] * hit.u + geom.normals[vidx.z] * hit.v).Normalized();
									normal = TransformDirection(local_normal, inst.inv_transform).Normalized();
								}
								else
								{
									Triangle const& tri = inst.blas->triangles[hit.tri_idx];
									Vector3 local_normal = Vector3::Cross(tri.v1 - tri.v0, tri.v2 - tri.v0).Normalized();
									normal = TransformDirection(local_normal, inst.inv_transform).Normalized();
								}

								if (normal.Dot(-ray.direction) < 0.0f)
								{
									normal = -normal;
								}

								Vector2 uv(0.0f, 0.0f);
								if (!geom.uvs.empty())
									uv = geom.uvs[vidx.x] * bw + geom.uvs[vidx.y] * hit.u + geom.uvs[vidx.z] * hit.v;

								Uint32 mat_id = blas_material_ids[blas_idx];
								Vector3 albedo(0.9f, 0.9f, 0.9f);
								if (mat_id < static_cast<Uint32>(scene->materials.size()))
								{
									Material const& mat = scene->materials[mat_id];
									albedo = mat.base_color;
									if (mat.diffuse_tex_id >= 0 && !geom.uvs.empty())
									{
										Vector3 texel = BilinearRepeat.Sample<Vector3>(textures[mat.diffuse_tex_id], uv);
										albedo = texel * mat.base_color;
									}
								}

								throughput = throughput * albedo;
								Vector3 hit_pos = ray.origin + ray.direction * hit.t + normal * 1e-4f;
								ray = Ray(hit_pos, CosineSampleHemisphere(normal, RandFloat(rng), RandFloat(rng)));
							}

							framebuffer(y, x) = ToDisplay(radiance);
						}
					}
				}));
			}
		}

		for (auto& f : futures)
		{
			f.get();
		}

		render_time_ms = timer.ElapsedInSeconds() * 1000.0f;
		AMBER_INFO_LOG("Frame %u: %.2f ms", frame_index, render_time_ms);
		++frame_index;
	}

	void CpuPathTracer::OnResize(Uint32 w, Uint32 h)
	{
		width = w;
		height = h;
		framebuffer.Resize(height, width);
		frame_index = 0;
	}

	void CpuPathTracer::WriteFramebuffer(Char const* outfile)
	{
		WriteImageToFile(ImageFormat::PNG, outfile, width, height, framebuffer.Data(), width * 4);
		AMBER_INFO_LOG("Wrote framebuffer to %s", outfile);
	}
}
