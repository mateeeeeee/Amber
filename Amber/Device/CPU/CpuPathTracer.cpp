#include "CpuPathTracer.h"
#include "PathTracingUtils.h"
#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include "Core/Log.h"
#include "Math/MathCommon.h"
#include "Utilities/ImageUtil.h"
#include "Utilities/Timer.h"
#include "Utilities/EnvVars.h"
#include "ImGui/imgui.h"
#include "BVH/Traversal.h"
#include <cmath>

namespace amber
{
	static constexpr Uint32 CPU_PT_DEFAULT_max_depth = 3;
	static constexpr Uint32 CPU_PT_DEFAULT_tile_size = 16;

	CpuPathTracer::CpuPathTracer(Uint32 width, Uint32 height, PathTracerConfig const& config, std::unique_ptr<Scene>&& _scene)
		: width(width), height(height),
		  max_depth(GetEnvVar("AMBER_max_depth", (Int)CPU_PT_DEFAULT_max_depth)),
		  tile_size(GetEnvVar("AMBER_tile_size", (Int)CPU_PT_DEFAULT_tile_size)),
		  scene(std::move(_scene)), framebuffer(height, width), accumulation_buffer(height, width)
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

		lights = scene->lights;

		Uint32 directional_light_count = 0;
		for (Light const& l : lights)
		{
			if (l.type == LightType::Directional) ++directional_light_count;
		}

		if (directional_light_count == 0)
		{
			Light& default_light    = lights.emplace_back();
			default_light.type      = LightType::Directional;
			default_light.color     = Vector3(8.0f, 8.0f, 8.0f);
			default_light.direction = Vector3(0.0f, -1.0f, 0.1f).Normalized();
		}

		bvh_tlas_stats = ComputeStats(tlas.bvh);
		bvh_blas_stats.clear();
		for (BLAS const& blas : blas_list)
		{
			bvh_blas_stats.push_back(ComputeStats(blas.bvh));
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
			accumulation_buffer.Clear(Vector3(0.0f, 0.0f, 0.0f));
		}

		Timer<std::chrono::microseconds> render_timer;
		Vector3 origin = camera.GetPosition();
		Vector3 U, V, W;
		camera.GetFrame(U, V, W);

		Float fov_rad = camera.GetFovY() * (3.14159265359f / 180.0f);
		Float tan_half_fov = std::tan(fov_rad * 0.5f);
		Float aspect = camera.GetAspectRatio();

		Uint32 tiles_x = (width + tile_size - 1) / tile_size;
		Uint32 tiles_y = (height + tile_size - 1) / tile_size;

		std::vector<std::future<void>> futures;
		futures.reserve(tiles_x * tiles_y);
		for (Uint32 ty = 0; ty < tiles_y; ++ty)
		{
			for (Uint32 tx = 0; tx < tiles_x; ++tx)
			{
				futures.push_back(g_ThreadPool.Submit([this, tx, ty, origin, U, V, W, tan_half_fov, aspect]()
				{
					Uint32 x_begin = tx * tile_size;
					Uint32 y_begin = ty * tile_size;
					Uint32 x_end = std::min(x_begin + tile_size, width);
					Uint32 y_end = std::min(y_begin + tile_size, height);

					for (Uint32 y = y_begin; y < y_end; ++y)
					{
						for (Uint32 x = x_begin; x < x_end; ++x)
						{
							Float ndc_x = (2.0f * (x + 0.5f) / width - 1.0f) * aspect * tan_half_fov;
							Float ndc_y = (1.0f - 2.0f * (y + 0.5f) / height) * tan_half_fov;

							Uint32 rng = PcgHash(x + PcgHash(y + PcgHash(frame_index)));

							Ray ray(origin, (W + U * ndc_x + V * ndc_y).Normalized());

						if (bvh_heatmap_enabled)
						{
							Uint32 count = 0;
							if (bvh_heatmap_mode == 0)
							{
								count = CountTraversalSteps(tlas.bvh, ray);
							}
							else if (bvh_heatmap_mode == 1)
							{
								count = CountPrimTests(tlas.bvh, ray);
							}
							else
							{
								HitInfo hit{};
								Ray r = ray;
								Bool found = Intersect(tlas, r, hit);
								Float t = found ? (hit.t / 100.0f) : 1.0f;
								t = std::min(t, 1.0f);
								Vector3 color(t * t, t * (1.0f - t) * 2.0f, (1.0f - t) * (1.0f - t));
								if (bvh_heatmap_blend)
								{
									accumulation_buffer(y, x) += Vector3(color.x, color.y, color.z);
									framebuffer(y, x) = ToDisplay(accumulation_buffer(y, x) / Float(frame_index + 1));
								}
								else
								{
									framebuffer(y, x) = RGBA8(
										static_cast<Uint8>(std::min(color.x * 255.0f, 255.0f)),
										static_cast<Uint8>(std::min(color.y * 255.0f, 255.0f)),
										static_cast<Uint8>(std::min(color.z * 255.0f, 255.0f)),
										255);
								}
								continue;
							}

							Float  t = std::min(static_cast<Float>(count) / static_cast<Float>(bvh_heatmap_max_steps), 1.0f);
							Vector3 color(t * t, t * (1.0f - t) * 2.0f, (1.0f - t) * (1.0f - t));
							framebuffer(y, x) = RGBA8(
								static_cast<Uint8>(std::min(color.x * 255.0f, 255.0f)),
								static_cast<Uint8>(std::min(color.y * 255.0f, 255.0f)),
								static_cast<Uint8>(std::min(color.z * 255.0f, 255.0f)),
								255);
							continue;
						}
							Vector3 throughput(1.0f, 1.0f, 1.0f);
							Vector3 radiance(0.0f, 0.0f, 0.0f);

							for (Uint32 depth = 0; depth < max_depth; ++depth)
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
								Vector3 emissive(0.0f, 0.0f, 0.0f);
								if (mat_id < static_cast<Uint32>(scene->materials.size()))
								{
									Material const& mat = scene->materials[mat_id];
									albedo = mat.base_color;
									if (mat.diffuse_tex_id >= 0 && !geom.uvs.empty())
									{
										Vector3 texel = BilinearRepeat.Sample<Vector3>(textures[mat.diffuse_tex_id], uv);
										albedo = texel * mat.base_color;
									}
									emissive = mat.emissive_color;
									if (mat.emissive_tex_id >= 0 && !geom.uvs.empty())
									{
										Vector3 tex_emissive = BilinearRepeat.Sample<Vector3>(textures[mat.emissive_tex_id], uv);
										emissive = emissive * tex_emissive;
									}
								}

								Vector3 hit_pos = ray.origin + ray.direction * hit.t + normal * 1e-4f;

								radiance += throughput * emissive;
								radiance += throughput * albedo * SampleDirectLight(tlas, lights, hit_pos, normal, rng);

								throughput = throughput * albedo;
								ray = Ray(hit_pos, CosineSampleHemisphere(normal, RandFloat(rng), RandFloat(rng)));
							}

							accumulation_buffer(y, x) += radiance;
							framebuffer(y, x) = ToDisplay(accumulation_buffer(y, x) / Float(frame_index + 1));
						}
					}
				}));
			}
		}

		for (auto& f : futures)
		{
			f.get();
		}

		render_time_ms = render_timer.ElapsedInSeconds() * 1000.0f;
		AMBER_INFO_LOG("Frame %u: %.2f ms", frame_index, render_time_ms);
		++frame_index;
	}

	void CpuPathTracer::OnResize(Uint32 w, Uint32 h)
	{
		width = w;
		height = h;
		framebuffer.Resize(height, width);
		accumulation_buffer.Resize(height, width);
		accumulation_buffer.Clear(Vector3(0.0f, 0.0f, 0.0f));
		frame_index = 0;
	}

	void CpuPathTracer::WriteFramebuffer(Char const* outfile)
	{
		WriteImageToFile(ImageFormat::PNG, outfile, width, height, framebuffer.Data(), width * 4);
		AMBER_INFO_LOG("Wrote framebuffer to %s", outfile);
	}

	void CpuPathTracer::LightEditorGUI()
	{
		Bool changed = false;
		Int light_index = 0;
		for (Light& light : lights)
		{
			std::string light_label = "Light " + std::to_string(light_index++);

			ImGui::PushID(light_index);
			ImGui::BeginChild(light_label.c_str(), ImVec2(0, 150), true, ImGuiWindowFlags_NoScrollbar);
			ImGui::Columns(2, nullptr, false);

			ImGui::Text("Light %d", light_index);
			ImGui::NextColumn();
			const Char* light_types[] = { "Directional", "Point" };
			ImGui::Combo("Type", (int*)&light.type, light_types, IM_ARRAYSIZE(light_types));
			ImGui::NextColumn();

			ImGui::Text("Color");
			ImGui::NextColumn();
			changed |= ImGui::ColorEdit3("##Color", &light.color.x, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
			ImGui::NextColumn();

			if (light.type == LightType::Directional)
			{
				ImGui::Text("Sun Elevation");
				ImGui::NextColumn();
				static Float sun_elevation = 75.0f;
				changed |= ImGui::SliderFloat("##Elevation", &sun_elevation, -90.0f, 90.0f);
				ImGui::NextColumn();

				ImGui::Text("Sun Azimuth");
				ImGui::NextColumn();
				static Float sun_azimuth = 260.0f;
				changed |= ImGui::SliderFloat("##Azimuth", &sun_azimuth, 0.0f, 360.0f);

				Vector3 dir = ConvertElevationAndAzimuthToDirection(sun_elevation, sun_azimuth);
				light.direction = Vector3(-dir.x, -dir.y, -dir.z);
			}
			else if (light.type == LightType::Point)
			{
				ImGui::Text("Position");
				ImGui::NextColumn();
				changed |= ImGui::InputFloat3("##Position", &light.position.x);
			}

			ImGui::Columns(1);
			ImGui::EndChild();
			ImGui::PopID();
			ImGui::Separator();
		}

		if (changed)
		{
			frame_index = 0;
			accumulation_buffer.Clear(Vector3(0.0f, 0.0f, 0.0f));
		}
	}

	void CpuPathTracer::BVHDebugGUI()
	{
		ImGui::Checkbox("BVH Heatmap", &bvh_heatmap_enabled);
		if (bvh_heatmap_enabled)
		{
			Char const* heatmap_modes[] = { "Traversal Steps", "Prim Tests", "First Hit Distance" };
			ImGui::SetNextItemWidth(160.0f);
			ImGui::Combo("Mode", &bvh_heatmap_mode, heatmap_modes, IM_ARRAYSIZE(heatmap_modes));
			if (bvh_heatmap_mode != 2)
			{
				ImGui::SameLine();
				ImGui::SetNextItemWidth(120.0f);
				ImGui::SliderInt("Max Steps", &bvh_heatmap_max_steps, 1, 256);
			}
			ImGui::Checkbox("Blend", &bvh_heatmap_blend);
			ImGui::TextDisabled("blue = few, green = mid, red = many");
		}

		ImGui::Separator();

		auto ShowStats = [](BVHStats const& s, Char const* label)
		{
			if (ImGui::TreeNode(label))
			{
				ImGui::Columns(2, nullptr, false);
				auto Row = [](Char const* name, auto val)
				{
					ImGui::TextUnformatted(name); ImGui::NextColumn();
					ImGui::Text("%s", std::to_string(val).c_str()); ImGui::NextColumn();
				};
				auto RowF = [](Char const* name, Float val)
				{
					ImGui::TextUnformatted(name); ImGui::NextColumn();
					ImGui::Text("%.4f", val); ImGui::NextColumn();
				};
				Row("Nodes",              s.node_count);
				Row("Leaves",             s.leaf_count);
				Row("Internal",           s.internal_count);
				Row("Max depth",          s.max_depth);
				Row("Only-leaf internals",s.nodes_only_leaves);
				Row("Only-int internals", s.nodes_only_internal);
				Row("Min leaf prims",     s.min_leaf_prims);
				Row("Max leaf prims",     s.max_leaf_prims);
				RowF("Avg leaf prims",    s.avg_leaf_prims);
				RowF("SAH cost",          s.sah_cost);
				RowF("Total SA",          s.total_sa);
				RowF("Leaf SA total",     s.leaf_sa_total);
				RowF("Leaf SA avg",       s.leaf_sa_avg);
				RowF("Leaf SA min",       s.leaf_sa_min);
				RowF("Leaf SA max",       s.leaf_sa_max);
				RowF("Total volume",      s.total_volume);
				RowF("Leaf vol total",    s.leaf_volume_total);
				RowF("Leaf vol avg",      s.leaf_volume_avg);
				RowF("Leaf vol min",      s.leaf_volume_min);
				RowF("Leaf vol max",      s.leaf_volume_max);
				ImGui::Columns(1);
				ImGui::TreePop();
			}
		};

		ShowStats(bvh_tlas_stats, "TLAS");
		for (Uint32 i = 0; i < static_cast<Uint32>(bvh_blas_stats.size()); i++)
		{
			std::string label = "BLAS " + std::to_string(i);
			ShowStats(bvh_blas_stats[i], label.c_str());
		}
	}
}
