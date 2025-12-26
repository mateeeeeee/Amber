#define CGLTF_IMPLEMENTATION
#include <unordered_map>
#include "Scene.h"
#include "Core/Log.h"
#include "Math/MathCommon.h"
#include "pbrtParser/Scene.h"
#include "tinyobjloader/tiny_obj_loader.h"
#include "cgltf/cgltf.h"
#include "tinyusdz.hh"
#include "tydra/render-data.hh"
#include "tydra/scene-access.hh"
#include "io-util.hh"


namespace amber
{
	enum class SceneFormat : Uint8
	{
		OBJ,
		GLTF,
		GLB,
		PBRT,
		PBF,
		USD,
		USDA,
		USDC,
		USDZ,
		Unknown
	};

	SceneFormat GetSceneFormat(std::string_view scene_file)
	{
		if (scene_file.ends_with(".pbrt")) return SceneFormat::PBRT;
		else if (scene_file.ends_with(".pbf")) return SceneFormat::PBF;
		else if (scene_file.ends_with(".obj")) return SceneFormat::OBJ;
		else if (scene_file.ends_with(".gltf")) return SceneFormat::GLTF;
		else if (scene_file.ends_with(".glb")) return SceneFormat::GLB;
		else if (scene_file.ends_with(".usd")) return SceneFormat::USD;
		else if (scene_file.ends_with(".usda")) return SceneFormat::USDA;
		else if (scene_file.ends_with(".usdc")) return SceneFormat::USDC;
		else if (scene_file.ends_with(".usdz")) return SceneFormat::USDZ;
		else return SceneFormat::Unknown;
	}
	
	namespace 
	{
		Int32 LoadPBRTTexture(
			Scene& scene,
			pbrt::Texture::SP const& texture,
			std::string_view pbrt_base_dir,
			std::unordered_map<pbrt::Texture::SP, Uint32>& pbrt_textures)
		{
			auto fnd = pbrt_textures.find(texture);
			if (pbrt_textures.contains(texture))
			{
				return pbrt_textures[texture];
			}

			if (auto t = texture->as<pbrt::ImageTexture>())
			{
				std::string path = std::string(pbrt_base_dir) + t->fileName;
				const Uint32 id = scene.textures.size();
				pbrt_textures[texture] = id;
				scene.textures.emplace_back(path.c_str(), true);
				return id;
			}
			else if (auto t = texture->as<pbrt::ConstantTexture>())
			{
				AMBER_WARN_LOG("LoadPBRTTexture for ConstantTexture! This case should be covered in LoadPBRTMaterial!");
			}
			else if (auto t = texture->as<pbrt::CheckerTexture>())
			{
			}
			else if (auto t = texture->as<pbrt::MixTexture>())
			{
			}
			else if (auto t = texture->as<pbrt::ScaleTexture>())
			{
			}

			return -1;
		}

		Uint32 LoadPBRTMaterials(Scene& scene,
			pbrt::Material::SP const& mat,
			std::map<std::string, pbrt::Texture::SP> const& texture_overrides,
			std::string_view pbrt_base_dir,
			std::unordered_map<pbrt::Material::SP, Uint32>& pbrt_materials,
			std::unordered_map<pbrt::Texture::SP, Uint32>& pbrt_textures)
		{
			if (pbrt_materials.contains(mat))
			{
				return pbrt_materials[mat];
			}

			Material material{};
			if (auto m = mat->as<pbrt::DisneyMaterial>())
			{
				material.anisotropy = m->anisotropic;
				material.clearcoat = m->clearCoat;
				material.clearcoat_gloss = m->clearCoatGloss;
				material.base_color = Vector3(m->color.x, m->color.y, m->color.z);
				material.ior = m->eta;
				material.metallic = m->metallic;
				material.roughness = m->roughness;
				material.sheen = m->sheen;
				material.sheen_tint = m->sheenTint;
				material.specular_tint = m->specularTint;
				material.specular_transmission = m->specTrans;
				material.specular = 0.0f;
			}
			else if (auto m = mat->as<pbrt::PlasticMaterial>())
			{
				material.base_color = Vector3(m->kd.x, m->kd.y, m->kd.z);
				if (m->map_kd) 
				{
					if (auto const_tex = m->map_kd->as<pbrt::ConstantTexture>())
					{
						material.base_color = Vector3(const_tex->value.x, const_tex->value.y, const_tex->value.z);
					}
					else 
					{
						Int32 tex_id = LoadPBRTTexture(scene, m->map_kd, pbrt_base_dir, pbrt_textures);
						material.diffuse_tex_id = tex_id;
					}
				}
				if (m->map_bump)
				{
					Int32 tex_id = LoadPBRTTexture(scene, m->map_bump, pbrt_base_dir, pbrt_textures);
					material.normal_tex_id = tex_id;
				}

				Vector3 ks(m->ks.x, m->ks.y, m->ks.z);
				material.specular = ks.x * 0.2126f +  0.7152f * ks.y + 0.0722f * ks.z;
				material.roughness = m->roughness;
			}
			else if (auto m = mat->as<pbrt::MatteMaterial>())
			{
				material.base_color = Vector3(m->kd.x, m->kd.y, m->kd.z);
				if (m->map_kd) 
				{
					if (auto const_tex = m->map_kd->as<pbrt::ConstantTexture>())
					{
						material.base_color = Vector3(const_tex->value.x, const_tex->value.y, const_tex->value.z);
					}
					else 
					{
						Int32 tex_id = LoadPBRTTexture(scene, m->map_kd, pbrt_base_dir, pbrt_textures);
						material.diffuse_tex_id = tex_id;
					}
				}
				if (m->map_bump)
				{
					Int32 tex_id = LoadPBRTTexture(scene, m->map_bump, pbrt_base_dir, pbrt_textures);
					material.normal_tex_id = tex_id;
				}
			}
			else if (auto m = mat->as<pbrt::SubstrateMaterial>())
			{
				material.base_color = Vector3(m->kd.x, m->kd.y, m->kd.z);
				if (m->map_kd) 
				{
					if (auto const_tex = m->map_kd->as<pbrt::ConstantTexture>())
					{
						material.base_color = Vector3(const_tex->value.x, const_tex->value.y, const_tex->value.z);
					}
					else 
					{
						Int32 tex_id = LoadPBRTTexture(scene, m->map_kd, pbrt_base_dir, pbrt_textures);
						material.diffuse_tex_id = tex_id;
					}
				}
				if (m->map_bump)
				{
					Int32 tex_id = LoadPBRTTexture(scene, m->map_bump, pbrt_base_dir, pbrt_textures);
					material.normal_tex_id = tex_id;
				}

				Vector3 ks(m->ks.x, m->ks.y, m->ks.z);
				material.specular = ks.x * 0.2126f + 0.7152f * ks.y + 0.0722f * ks.z;
				material.roughness = 1.0f;
				material.clearcoat = 1.0f;
				material.clearcoat_gloss = material.specular;
			}
			else if (auto m = mat->as<pbrt::MetalMaterial>())
			{
				// Metallic property
				material.metallic = 1.0f;

				// Roughness
				material.roughness = m->roughness;
				if (m->remapRoughness) 
				{
					material.roughness = sqrt(m->roughness); // Example remapping
				}

				auto ComputeReflectance = [](const pbrt::vec3f& eta, const pbrt::vec3f& k)
					{
						pbrt::vec3f numerator = (eta - pbrt::vec3f(1.0f)) * (eta - pbrt::vec3f(1.0f)) + k * k;
						pbrt::vec3f denominator = (eta + pbrt::vec3f(1.0f)) * (eta + pbrt::vec3f(1.0f)) + k * k;
						return Vector3(numerator.x / denominator.x, numerator.y / denominator.y, numerator.z / denominator.z); 
					};
				material.base_color = ComputeReflectance(m->eta, m->k);

				float avg_eta = (m->eta.x + m->eta.y + m->eta.z) / 3.0f;
				float avg_k = (m->k.x + m->k.y + m->k.z) / 3.0f;
				float avg_reflectance = avg_eta / (avg_eta + avg_k);
				material.specular_tint = 1.0f - avg_reflectance;

				if (m->uRoughness != m->vRoughness)
				{
					material.anisotropy = fabs(m->uRoughness - m->vRoughness) / (m->uRoughness + m->vRoughness);
				}
				else 
				{
					material.anisotropy = 0.0f;
				}

				material.clearcoat = 0.0f;
				material.sheen = 0.0f;
				material.ior = (m->eta.x + m->eta.y + m->eta.z) / 3.0f;

				if (m->map_bump)
				{
					Int32 tex_id = LoadPBRTTexture(scene, m->map_bump, pbrt_base_dir, pbrt_textures);
					material.normal_tex_id = tex_id;
				}
			}
			else if (auto m = mat->as<pbrt::GlassMaterial>())
			{
				material.anisotropy = 0.0f;
				material.clearcoat = 0.0f;
				material.clearcoat_gloss = 1.0f;
				material.base_color = Vector3(m->kr.x, m->kr.y, m->kr.z);

				material.ior = m->index;
				material.metallic = 0.0f;
				material.roughness = 0.0f;
				material.sheen = 0.0f;
				material.sheen_tint = 0.0f;
				material.specular_transmission = 1.0f;
				material.specular = 0.0f;

				float kr_intensity = (m->kr.x + m->kr.y + m->kr.z) / 3.0f;
				if (kr_intensity > 0.0f) 
				{
					auto min = []<typename T>(T a, T b) { return a < b ? a : b; };
					material.specular_tint = 1.0f - min(m->kr.x, min(m->kr.y, m->kr.z)) / kr_intensity;
				}
				else 
				{
					material.specular_tint = 0.0f;
				}

				material.base_color.x *= (1.0f - m->kt.x);
				material.base_color.y *= (1.0f - m->kt.y);
				material.base_color.z *= (1.0f - m->kt.z);
			}
			Uint32 mat_id = scene.materials.size();
			pbrt_materials[mat] = mat_id;
			scene.materials.push_back(material);
			return mat_id;
		}

		std::unique_ptr<Scene> ConvertPBRTScene(std::shared_ptr<pbrt::Scene> const& pbrt_scene, std::string_view scene_file)
		{
			pbrt_scene->makeSingleLevel();
			std::unique_ptr<Scene> scene = std::make_unique<Scene>();
			auto const& pbrt_world = pbrt_scene->world;
			for (auto const& pbrt_light : pbrt_world->lightSources)
			{
				if (auto pbrt_distant_light = pbrt_light->as<pbrt::DistantLightSource>())
				{

				}
				else if (auto pbrt_point_light = pbrt_light->as<pbrt::PointLightSource>())
				{
				}
				else
				{
					AMBER_WARN_LOG("Light source type %s not yet supported", pbrt_light->toString().c_str());
				}
			}

			// For PBRTv3 Each Mesh corresponds to a PBRT Object, consisting of potentially
			// multiple Shapes. This maps to a Mesh with multiple geometries, which can then be
			// instanced
			std::unordered_map<pbrt::Material::SP, Uint32>	pbrt_materials;
			std::unordered_map<pbrt::Texture::SP, Uint32>	pbrt_textures;
			std::unordered_map<std::string, Uint32>			pbrt_objects;
			for (auto const& instance : pbrt_world->instances)
			{
				Uint32 primitive_id = -1;
				if (!pbrt_objects.contains(instance->object->name))
				{
					std::vector<Uint32> material_ids;
					std::vector<Geometry> geometries;
					for (const auto& shape : instance->object->shapes)
					{
						if (pbrt::TriangleMesh::SP mesh = shape->as<pbrt::TriangleMesh>())
						{
							Uint32 material_id = -1;
							if (mesh->material)
							{
								material_id = LoadPBRTMaterials(*scene, mesh->material,
									mesh->textures,
									scene_file,
									pbrt_materials,
									pbrt_textures);
							}
							material_ids.push_back(material_id);

							Geometry geom{};
							geom.vertices.reserve(mesh->vertex.size());
							for (pbrt::vec3f const& v : mesh->vertex)
							{
								geom.vertices.emplace_back(v.x, v.y, v.z);
							}
	
							geom.indices.reserve(mesh->index.size());
							for (pbrt::vec3i const& i : mesh->index)
							{
								geom.indices.emplace_back(i.x, i.y, i.z);
							}

							geom.uvs.reserve(mesh->texcoord.size());
							for (pbrt::vec2f const& t : mesh->texcoord)
							{
								geom.uvs.emplace_back(t.x, t.y);
							}

							geom.normals.reserve(mesh->normal.size());
							for (pbrt::vec3f const& n : mesh->normal)
							{
								geom.normals.emplace_back(n.x, n.y, n.z);
							}
			
							geometries.push_back(geom);
						}
						else if (pbrt::Sphere::SP mesh = shape->as<pbrt::Sphere>())
						{
							AMBER_WARN_LOG("Sphere mesh encountered! Not supported yet.");
						}
						else if (pbrt::QuadMesh::SP mesh = shape->as<pbrt::QuadMesh>())
						{
							AMBER_WARN_LOG("Quad mesh encountered! Not supported yet.");
						}
						else
						{
							AMBER_WARN_LOG("Unsupported mesh type encountered!");
						}
					}

					if (geometries.empty()) 
					{
						AMBER_WARN_LOG("Skipping instance with unsupported geometry...");
						continue;
					}

					Uint32 const mesh_id = scene->meshes.size();
					scene->meshes.emplace_back(geometries);

					pbrt_objects[instance->object->name] = mesh_id;
				}
				primitive_id = pbrt_objects[instance->object->name];

				Vector4 v0 = Vector4(instance->xfm.l.vx.x, instance->xfm.l.vx.y, instance->xfm.l.vx.z, 0.f);
				Vector4 v1 = Vector4(instance->xfm.l.vy.x, instance->xfm.l.vy.y, instance->xfm.l.vy.z, 0.f);
				Vector4 v2 = Vector4(instance->xfm.l.vz.x, instance->xfm.l.vz.y, instance->xfm.l.vz.z, 0.f);
				Vector4 v3 = Vector4(instance->xfm.p.x, instance->xfm.p.y, instance->xfm.p.z, 1.f);
				Matrix transform(v0, v1, v2, v3);
				scene->instances.emplace_back(transform, primitive_id);

				return scene;
			}

			return scene;
		}

		std::unique_ptr<Scene> LoadObjScene(std::string_view scene_file, Float scale)
		{
			tinyobj::ObjReaderConfig reader_config{};
			tinyobj::ObjReader reader;
			if (!reader.ParseFromFile(std::string(scene_file), reader_config))
			{
				if (!reader.Error().empty())
				{
					AMBER_ERROR_LOG("TinyOBJ error: %s", reader.Error().c_str());
				}
				return nullptr;
			}
			if (!reader.Warning().empty())
			{
				AMBER_WARN_LOG("TinyOBJ warning: %s", reader.Warning().c_str());
			}

			std::string obj_base_dir = std::string(scene_file.substr(0, scene_file.rfind('/')));
			tinyobj::attrib_t const& attrib = reader.GetAttrib();
			std::vector<tinyobj::shape_t> const& shapes = reader.GetShapes();
			std::vector<tinyobj::material_t> const& materials = reader.GetMaterials();

			std::unique_ptr<Scene> obj_scene = std::make_unique<Scene>();

			Mesh mesh;
			std::vector<Uint32> material_ids;
			for (Uint64 s = 0; s < shapes.size(); s++)
			{
				tinyobj::mesh_t const& obj_mesh = shapes[s].mesh;
				if(obj_mesh.material_ids[0] >= 0) material_ids.push_back(obj_mesh.material_ids[0]);
				
				Geometry geometry{};
				Uint32 index_offset = 0;
				for (Uint64 f = 0; f < obj_mesh.num_face_vertices.size(); ++f) 
				{
					AMBER_ASSERT(obj_mesh.num_face_vertices[f] == 3);
					for (Uint64 v = 0; v < 3; v++)
					{
						tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
						tinyobj::real_t vx = attrib.vertices[3 * Uint64(idx.vertex_index) + 0] * scale;
						tinyobj::real_t vy = attrib.vertices[3 * Uint64(idx.vertex_index) + 1] * scale;
						tinyobj::real_t vz = attrib.vertices[3 * Uint64(idx.vertex_index) + 2] * scale * -1.0f;

						geometry.vertices.emplace_back(vx, vy, vz);

						if (idx.normal_index >= 0)
						{
							tinyobj::real_t nx = attrib.normals[3 * Uint64(idx.normal_index) + 0];
							tinyobj::real_t ny = attrib.normals[3 * Uint64(idx.normal_index) + 1];
							tinyobj::real_t nz = attrib.normals[3 * Uint64(idx.normal_index) + 2];
							geometry.normals.emplace_back(nx, ny, nz);
						}

						if (idx.texcoord_index >= 0)
						{
							tinyobj::real_t tx = attrib.texcoords[2 * Uint64(idx.texcoord_index) + 0];
							tinyobj::real_t ty = attrib.texcoords[2 * Uint64(idx.texcoord_index) + 1];

							geometry.uvs.emplace_back(tx, ty);
						}
					}
					geometry.indices.emplace_back(index_offset, index_offset + 1, index_offset + 2);
					index_offset += 3;
				}
				obj_scene->instances.emplace_back(Matrix::Identity, mesh.geometries.size());
				mesh.geometries.push_back(geometry);
				mesh.material_ids.push_back(obj_mesh.material_ids[0]);
			}
			obj_scene->meshes.push_back(std::move(mesh));

			std::unordered_map<std::string, Int32> texture_ids;
			for (auto const& m : materials)
			{
				Material material{};
				material.base_color = Vector3(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
				material.emissive_color = Vector3(m.emission[0], m.emission[1], m.emission[2]);
				material.specular = (m.specular[0] + m.specular[1] + m.specular[2]) / 3.0f;
				material.ior = m.ior;
				material.specular_transmission = m.dissolve < 1.0f ? 1.0f - m.dissolve : 0.0f;
				material.roughness = Clamp(1.0f - (m.shininess / 1000.0f), 0.0f, 1.0f);
				material.metallic = m.metallic;
				material.sheen = m.sheen;
				material.clearcoat = m.clearcoat_thickness;
				material.clearcoat_gloss = 1.0f - m.clearcoat_roughness;
				material.anisotropy = m.anisotropy;

				if (!m.diffuse_texname.empty()) 
				{
					if (!texture_ids.contains(m.diffuse_texname))
					{
						texture_ids[m.diffuse_texname] = obj_scene->textures.size();
						std::string texture_path = obj_base_dir + "/" + m.diffuse_texname;
						obj_scene->textures.emplace_back(texture_path.c_str(), true);
					}
					const Int32 id = texture_ids[m.diffuse_texname];
					material.diffuse_tex_id = id;
				}
				if (!m.normal_texname.empty())
				{
					if (!texture_ids.contains(m.normal_texname))
					{
						texture_ids[m.normal_texname] = obj_scene->textures.size();
						std::string texture_path = obj_base_dir + "/" + m.normal_texname;
						obj_scene->textures.emplace_back(texture_path.c_str(), false);
					}
					const Int32 id = texture_ids[m.normal_texname];
					material.normal_tex_id = id;
				}
				if (!m.emissive_texname.empty())
				{
					if (!texture_ids.contains(m.emissive_texname))
					{
						texture_ids[m.emissive_texname] = obj_scene->textures.size();
						std::string texture_path = obj_base_dir + "/" + m.emissive_texname;
						obj_scene->textures.emplace_back(texture_path.c_str(), false);
					}
					const Int32 id = texture_ids[m.emissive_texname];
					material.emissive_tex_id = id;
				}
				obj_scene->materials.push_back(material);
			}

			return obj_scene;
		}

		std::unique_ptr<Scene> LoadGltfScene(std::string_view scene_file, Float scale)
		{
			cgltf_options options{};
			cgltf_data* gltf_data = nullptr;
			
			cgltf_result result = cgltf_parse_file(&options, scene_file.data(), &gltf_data);
			if (result != cgltf_result_success)
			{
				AMBER_WARN_LOG("GLTF - Failed to load '%s'", scene_file.data());
				return nullptr;
			}
			result = cgltf_load_buffers(&options, gltf_data, scene_file.data());
			if (result != cgltf_result_success)
			{
				AMBER_WARN_LOG("GLTF - Failed to load buffers '%s'", scene_file.data());
				return nullptr;
			}

			std::string gltf_base_dir = std::string(scene_file.substr(0, scene_file.rfind('/')));
			std::unique_ptr<Scene> gltf_scene = std::make_unique<Scene>();

			std::unordered_map<std::string, Int32> texture_ids;
			gltf_scene->materials.reserve(gltf_data->materials_count);
			for (Uint32 i = 0; i < gltf_data->materials_count; ++i)
			{
				cgltf_material const& gltf_material = gltf_data->materials[i];
				Material& material = gltf_scene->materials.emplace_back();

				cgltf_pbr_metallic_roughness pbr_metallic_roughness = gltf_material.pbr_metallic_roughness;
				material.base_color.x = (Float)pbr_metallic_roughness.base_color_factor[0];
				material.base_color.y = (Float)pbr_metallic_roughness.base_color_factor[1];
				material.base_color.z = (Float)pbr_metallic_roughness.base_color_factor[2];
				material.metallic = (Float)pbr_metallic_roughness.metallic_factor;
				material.roughness = (Float)pbr_metallic_roughness.roughness_factor;
				material.emissive_color.x = (Float)gltf_material.emissive_factor[0];
				material.emissive_color.y = (Float)gltf_material.emissive_factor[1];
				material.emissive_color.z = (Float)gltf_material.emissive_factor[2];
				material.alpha_cutoff = (Float)gltf_material.alpha_cutoff;

				if (gltf_material.has_clearcoat)
				{
					cgltf_clearcoat const& gltf_clearcoat = gltf_material.clearcoat;
					material.clearcoat = gltf_clearcoat.clearcoat_factor;
					material.clearcoat_gloss = 1.0f - gltf_clearcoat.clearcoat_roughness_factor;
				}
				if (gltf_material.has_sheen)
				{
					cgltf_sheen const& gltf_sheen = gltf_material.sheen;
					material.sheen = 0.2126f * gltf_sheen.sheen_color_factor[0] +
									 0.7152f * gltf_sheen.sheen_color_factor[1] +
									 0.0722f * gltf_sheen.sheen_color_factor[2];

					Float average_color = (gltf_sheen.sheen_color_factor[0] +
										   gltf_sheen.sheen_color_factor[1] +
										   gltf_sheen.sheen_color_factor[2]) / 3.0f;

					material.sheen_tint = material.sheen > 0.0f ? (average_color / material.sheen) : 0.0f;
				}
				if (gltf_material.has_emissive_strength)
				{
					material.emissive_color.x *= gltf_material.emissive_strength.emissive_strength;
					material.emissive_color.y *= gltf_material.emissive_strength.emissive_strength;
					material.emissive_color.z *= gltf_material.emissive_strength.emissive_strength;
				}
				if (gltf_material.has_ior)
				{
					material.ior = gltf_material.ior.ior;
				}
				if (gltf_material.has_specular)
				{
					cgltf_specular const& gltf_specular = gltf_material.specular;

					material.specular = gltf_specular.specular_factor;
					material.specular_tint = 0.2126f * gltf_specular.specular_color_factor[0] +
											 0.7152f * gltf_specular.specular_color_factor[1] +
											 0.0722f * gltf_specular.specular_color_factor[2];

					Float average_color = (gltf_specular.specular_color_factor[0] +
										   gltf_specular.specular_color_factor[1] +
										   gltf_specular.specular_color_factor[2]) / 3.0f;

					material.specular_tint = material.specular_tint > 0.0f ? (average_color / material.specular_tint) : 0.0f;
				}
				if (gltf_material.has_transmission)
				{
					material.specular_transmission = gltf_material.transmission.transmission_factor;
				}

				if (cgltf_texture* texture = pbr_metallic_roughness.base_color_texture.texture)
				{
					cgltf_image* image = texture->image;
					if (!texture_ids.contains(image->uri))
					{
						texture_ids[image->uri] = gltf_scene->textures.size();
						std::string texture_path = gltf_base_dir + "/" + image->uri;
						gltf_scene->textures.emplace_back(texture_path.c_str(), true);
					}
					const Int32 id = texture_ids[image->uri];
					material.diffuse_tex_id = id;
				}

				if (cgltf_texture* texture = pbr_metallic_roughness.metallic_roughness_texture.texture)
				{
					cgltf_image* image = texture->image;
					if (!texture_ids.contains(image->uri))
					{
						texture_ids[image->uri] = gltf_scene->textures.size();
						std::string texture_path = gltf_base_dir + "/" + image->uri;
						gltf_scene->textures.emplace_back(texture_path.c_str(), true);
					}
					const Int32 id = texture_ids[image->uri];
					material.metallic_roughness_tex_id = id;
				}

				if (cgltf_texture* texture = gltf_material.normal_texture.texture)
				{
					cgltf_image* image = texture->image;
					if (!texture_ids.contains(image->uri))
					{
						texture_ids[image->uri] = gltf_scene->textures.size();
						std::string texture_path = gltf_base_dir + "/" + image->uri;
						gltf_scene->textures.emplace_back(texture_path.c_str(), true);
					}
					const Int32 id = texture_ids[image->uri];
					material.normal_tex_id = id;
				}

				if (cgltf_texture* texture = gltf_material.emissive_texture.texture)
				{
					cgltf_image* image = texture->image;
					if (!texture_ids.contains(image->uri))
					{
						texture_ids[image->uri] = gltf_scene->textures.size();
						std::string texture_path = gltf_base_dir + "/" + image->uri;
						gltf_scene->textures.emplace_back(texture_path.c_str(), true);
					}
					const Int32 id = texture_ids[image->uri];
					material.emissive_tex_id = id;
				}

			}

			std::unordered_map<cgltf_mesh const*, std::vector<Int32>> mesh_primitives_map; 
			Int32 primitive_count = 0;

			for (Uint32 i = 0; i < gltf_data->meshes_count; ++i)
			{
				cgltf_mesh const& gltf_mesh = gltf_data->meshes[i];
				std::vector<Int32>& primitives = mesh_primitives_map[&gltf_mesh];

				Mesh& mesh = gltf_scene->meshes.emplace_back();
				for (Uint32 j = 0; j < gltf_mesh.primitives_count; ++j)
				{
					auto const& gltf_primitive = gltf_mesh.primitives[j];
					AMBER_ASSERT(gltf_primitive.indices->count >= 0);

					Geometry& geometry = mesh.geometries.emplace_back();
					mesh.material_ids.push_back((Int32)(gltf_primitive.material - gltf_data->materials));

					geometry.indices.reserve(gltf_primitive.indices->count / 3);

					Uint32 triangle_cw[] = { 0, 1, 2 };
					Uint32 triangle_ccw[] = { 0, 2, 1 };
					Uint32* order = triangle_ccw;
					for (Uint64 i = 0; i < gltf_primitive.indices->count; i += 3)
					{
						Uint32 i0 = (Uint32)cgltf_accessor_read_index(gltf_primitive.indices, i + order[0]);
						Uint32 i1 = (Uint32)cgltf_accessor_read_index(gltf_primitive.indices, i + order[1]);
						Uint32 i2 = (Uint32)cgltf_accessor_read_index(gltf_primitive.indices, i + order[2]);
						geometry.indices.emplace_back(i0, i1, i2);
					}

					for (Uint32 k = 0; k < gltf_primitive.attributes_count; ++k)
					{
						cgltf_attribute const& gltf_attribute = gltf_primitive.attributes[k];
						std::string const& attr_name = gltf_attribute.name;

						auto ReadAttributeData = [&]<typename T>(std::vector<T>&stream, const Char* stream_name)
						{
							if (!attr_name.compare(stream_name))
							{
								stream.resize(gltf_attribute.data->count);
								for (Uint64 i = 0; i < gltf_attribute.data->count; ++i)
								{
									cgltf_accessor_read_float(gltf_attribute.data, i, &stream[i].x, sizeof(T) / sizeof(Float));
								}
							}
						};
						ReadAttributeData(geometry.vertices, "POSITION");
						ReadAttributeData(geometry.normals, "NORMAL");
						ReadAttributeData(geometry.uvs, "TEXCOORD_0");
					}
					primitives.push_back(primitive_count++);
				}
			}

			for (Uint64 i = 0; i < gltf_data->nodes_count; ++i)
			{
				cgltf_node const& gltf_node = gltf_data->nodes[i];

				Matrix local_to_world;
				cgltf_node_transform_world(&gltf_node, &local_to_world.m[0][0]);
				local_to_world *= Matrix::CreateScale(scale, scale, -scale);
				if (gltf_node.mesh)
				{
					for (Int32 primitive : mesh_primitives_map[gltf_node.mesh])
					{
						Instance& instance = gltf_scene->instances.emplace_back();
						instance.mesh_id = primitive;
						instance.transform = local_to_world;
					}
				}

				if (gltf_node.light)
				{
					cgltf_light const& gltf_light = *gltf_node.light;
					Vector3 translation, scale;
					Quaternion rotation;
					local_to_world.Decompose(translation, rotation, scale);

					Light& light = gltf_scene->lights.emplace_back();
					light.color.x = gltf_light.color[0] * gltf_light.intensity;
					light.color.y = gltf_light.color[1] * gltf_light.intensity;
					light.color.z = gltf_light.color[2] * gltf_light.intensity;
					light.position = Vector3(translation.x, translation.y, translation.z);
					Vector3 forward(0.0f, 0.0f, -1.0f);
					Vector3 direction = Vector3::Transform(forward, Matrix::CreateFromQuaternion(rotation));
					light.direction = Vector3(direction.x, direction.y, direction.z);

					switch (gltf_light.type)
					{
					case cgltf_light_type_directional: light.type = LightType::Directional; break;
					case cgltf_light_type_point:	   light.type = LightType::Point; break;
					case cgltf_light_type_spot:		   light.type = LightType::Spot; break;
					}
				}
			}

			return gltf_scene;
		}

		std::unique_ptr<Scene> LoadUsdScene(std::string_view scene_file, Float scale)
		{
			std::string warn, err;
			tinyusdz::Stage stage;
			Bool ret = tinyusdz::LoadUSDFromFile(std::string(scene_file), &stage, &warn, &err);
			if (!warn.empty())
			{
				AMBER_WARN_LOG("USD warning: %s", warn.c_str());
			}
			if (!ret)
			{
				AMBER_ERROR_LOG("USD - Failed to load '%s': %s", scene_file.data(), err.c_str());
				return nullptr;
			}

			tinyusdz::tydra::RenderScene render_scene;
			tinyusdz::tydra::RenderSceneConverter converter;
			tinyusdz::tydra::RenderSceneConverterEnv env(stage);

			env.mesh_config.triangulate = true;
			env.mesh_config.build_vertex_indices = true;
			env.mesh_config.compute_normals = true;
			std::string usd_base_dir = tinyusdz::io::GetBaseDir(std::string(scene_file));
			env.set_search_paths({usd_base_dir});
			env.scene_config.load_texture_assets = true;

			if (!converter.ConvertToRenderScene(env, &render_scene))
			{
				AMBER_ERROR_LOG("USD - Failed to convert scene: %s", converter.GetError().c_str());
				return nullptr;
			}

			std::unique_ptr<Scene> usd_scene = std::make_unique<Scene>();
			std::unordered_map<std::string, Int32> texture_ids;
			for (auto const& usd_material : render_scene.materials)
			{
				Material material{};

				auto const& shader = usd_material.surfaceShader;
				material.base_color = Vector3(
					shader.diffuseColor.value[0],
					shader.diffuseColor.value[1],
					shader.diffuseColor.value[2]
				);
				material.emissive_color = Vector3(
					shader.emissiveColor.value[0],
					shader.emissiveColor.value[1],
					shader.emissiveColor.value[2]
				);
				material.metallic = shader.metallic.value;
				material.roughness = shader.roughness.value;
				material.clearcoat = shader.clearcoat.value;
				material.clearcoat_gloss = 1.0f - shader.clearcoatRoughness.value;
				material.ior = shader.ior.value;
				material.specular_transmission = 1.0f - shader.opacity.value;
				material.alpha_cutoff = shader.opacityThreshold.value;

				if (shader.diffuseColor.is_texture() && shader.diffuseColor.texture_id >= 0)
				{
					Int32 tex_idx = shader.diffuseColor.texture_id;
					if (tex_idx < (Int32)render_scene.textures.size())
					{
						auto const& uv_tex = render_scene.textures[tex_idx];
						if (uv_tex.texture_image_id >= 0 && uv_tex.texture_image_id < (Int64)render_scene.images.size())
						{
							auto const& tex_image = render_scene.images[uv_tex.texture_image_id];
							std::string tex_path = tex_image.asset_identifier;
							if (!tex_path.empty())
							{
								if (!texture_ids.contains(tex_path))
								{
									texture_ids[tex_path] = usd_scene->textures.size();
									std::string full_path = usd_base_dir + "/" + tex_path;
									usd_scene->textures.emplace_back(full_path.c_str(), true);
								}
								material.diffuse_tex_id = texture_ids[tex_path];
							}
						}
					}
				}

				if (shader.normal.is_texture() && shader.normal.texture_id >= 0)
				{
					Int32 tex_idx = shader.normal.texture_id;
					if (tex_idx < (Int32)render_scene.textures.size())
					{
						auto const& uv_tex = render_scene.textures[tex_idx];
						if (uv_tex.texture_image_id >= 0 && uv_tex.texture_image_id < (Int64)render_scene.images.size())
						{
							auto const& tex_image = render_scene.images[uv_tex.texture_image_id];
							std::string tex_path = tex_image.asset_identifier;
							if (!tex_path.empty())
							{
								if (!texture_ids.contains(tex_path))
								{
									texture_ids[tex_path] = usd_scene->textures.size();
									std::string full_path = usd_base_dir + "/" + tex_path;
									usd_scene->textures.emplace_back(full_path.c_str(), false);
								}
								material.normal_tex_id = texture_ids[tex_path];
							}
						}
					}
				}

				if (shader.metallic.is_texture() && shader.metallic.texture_id >= 0)
				{
					Int32 tex_idx = shader.metallic.texture_id;
					if (tex_idx < (Int32)render_scene.textures.size())
					{
						auto const& uv_tex = render_scene.textures[tex_idx];
						if (uv_tex.texture_image_id >= 0 && uv_tex.texture_image_id < (Int64)render_scene.images.size())
						{
							auto const& tex_image = render_scene.images[uv_tex.texture_image_id];
							std::string tex_path = tex_image.asset_identifier;
							if (!tex_path.empty())
							{
								if (!texture_ids.contains(tex_path))
								{
									texture_ids[tex_path] = usd_scene->textures.size();
									std::string full_path = usd_base_dir + "/" + tex_path;
									usd_scene->textures.emplace_back(full_path.c_str(), false);
								}
								material.metallic_roughness_tex_id = texture_ids[tex_path];
							}
						}
					}
				}

				if (shader.emissiveColor.is_texture() && shader.emissiveColor.texture_id >= 0)
				{
					Int32 tex_idx = shader.emissiveColor.texture_id;
					if (tex_idx < (Int32)render_scene.textures.size())
					{
						auto const& uv_tex = render_scene.textures[tex_idx];
						if (uv_tex.texture_image_id >= 0 && uv_tex.texture_image_id < (Int64)render_scene.images.size())
						{
							auto const& tex_image = render_scene.images[uv_tex.texture_image_id];
							std::string tex_path = tex_image.asset_identifier;
							if (!tex_path.empty())
							{
								if (!texture_ids.contains(tex_path))
								{
									texture_ids[tex_path] = usd_scene->textures.size();
									std::string full_path = usd_base_dir + "/" + tex_path;
									usd_scene->textures.emplace_back(full_path.c_str(), false);
								}
								material.emissive_tex_id = texture_ids[tex_path];
							}
						}
					}
				}
				usd_scene->materials.push_back(material);
			}

			if (usd_scene->materials.empty())
			{
				usd_scene->materials.emplace_back();
			}

			for (Uint32 mesh_idx = 0; mesh_idx < render_scene.meshes.size(); ++mesh_idx)
			{
				auto const& usd_mesh = render_scene.meshes[mesh_idx];
				Mesh mesh;
				Geometry geometry{};
				geometry.vertices.reserve(usd_mesh.points.size());
				for (auto const& p : usd_mesh.points)
				{
					geometry.vertices.emplace_back(p[0] * scale, p[1] * scale, p[2] * scale * -1.0f);
				}

				auto const& indices = usd_mesh.faceVertexIndices();
				geometry.indices.reserve(indices.size() / 3);
				for (Uint64 i = 0; i + 2 < indices.size(); i += 3)
				{
					geometry.indices.emplace_back(indices[i], indices[i + 2], indices[i + 1]);
				}

				if (!usd_mesh.normals.empty())
				{
					auto const& norm_data = usd_mesh.normals.get_data();
					Uint64 num_normals = usd_mesh.normals.vertex_count();
					geometry.normals.reserve(num_normals);

					const float* norm_ptr = reinterpret_cast<const float*>(norm_data.data());
					for (Uint64 i = 0; i < num_normals; ++i)
					{
						geometry.normals.emplace_back(norm_ptr[i * 3], norm_ptr[i * 3 + 1], norm_ptr[i * 3 + 2]);
					}
				}

				auto tex_it = usd_mesh.texcoords.find(0);
				if (tex_it != usd_mesh.texcoords.end())
				{
					auto const& uv_attr = tex_it->second;
					auto const& uv_data = uv_attr.get_data();
					Uint64 num_uvs = uv_attr.vertex_count();
					geometry.uvs.reserve(num_uvs);

					const float* uv_ptr = reinterpret_cast<const float*>(uv_data.data());
					for (Uint64 i = 0; i < num_uvs; ++i)
					{
						geometry.uvs.emplace_back(uv_ptr[i * 2], uv_ptr[i * 2 + 1]);
					}
				}

				mesh.geometries.push_back(std::move(geometry));
				Int32 mat_id = usd_mesh.material_id >= 0 ? usd_mesh.material_id : 0;
				mesh.material_ids.push_back(mat_id);
				usd_scene->meshes.push_back(std::move(mesh));
			}

			std::function<void(tinyusdz::tydra::Node const&)> ProcessNode = [&](tinyusdz::tydra::Node const& node)
			{
				if (node.nodeType == tinyusdz::tydra::NodeType::Mesh && node.id >= 0)
				{
					Instance instance;
					instance.mesh_id = node.id;

					auto const& m = node.global_matrix;
					Matrix transform(
						Vector4((Float)m.m[0][0], (Float)m.m[0][1], (Float)m.m[0][2], (Float)m.m[0][3]),
						Vector4((Float)m.m[1][0], (Float)m.m[1][1], (Float)m.m[1][2], (Float)m.m[1][3]),
						Vector4((Float)m.m[2][0], (Float)m.m[2][1], (Float)m.m[2][2], (Float)m.m[2][3]),
						Vector4((Float)m.m[3][0], (Float)m.m[3][1], (Float)m.m[3][2], (Float)m.m[3][3])
					);

					transform *= Matrix::CreateScale(scale, scale, -scale);
					instance.transform = transform;
					usd_scene->instances.push_back(instance);
				}
				for (auto const& child : node.children)
				{
					ProcessNode(child);
				}
			};

			for (auto const& root_node : render_scene.nodes)
			{
				ProcessNode(root_node);
			}

			if (usd_scene->instances.empty() && !usd_scene->meshes.empty())
			{
				for (Uint64 i = 0; i < usd_scene->meshes.size(); ++i)
				{
					Instance instance;
					instance.mesh_id = i;
					instance.transform = Matrix::CreateScale(scale, scale, -scale);
					usd_scene->instances.push_back(instance);
				}
			}

			return usd_scene;
		}
	}

	std::unique_ptr<Scene> LoadScene(Char const* _scene_file, Char const* _environment_texture, Float scale)
	{
		std::string_view scene_file(_scene_file);
		std::string_view environment_texture(_environment_texture);
		SceneFormat scene_format = GetSceneFormat(scene_file);

		std::unique_ptr<Scene> scene = nullptr;
		switch (scene_format)
		{
		case SceneFormat::OBJ:
		{
			scene = LoadObjScene(scene_file, scale);
		}
		break;
		case SceneFormat::GLTF:
		{
			scene = LoadGltfScene(scene_file, scale);
		}
		break;
		case SceneFormat::PBRT:
		{
			std::shared_ptr<pbrt::Scene> pbrt_scene = pbrt::importPBRT(_scene_file);
			scene = ConvertPBRTScene(pbrt_scene, scene_file);
		}
		break;
		case SceneFormat::PBF:
		{
			std::shared_ptr<pbrt::Scene> pbrt_scene = pbrt::Scene::loadFrom(_scene_file);
			scene = ConvertPBRTScene(pbrt_scene, scene_file);
		}
		break;
		case SceneFormat::USD:
		case SceneFormat::USDA:
		case SceneFormat::USDC:
		case SceneFormat::USDZ:
		{
			scene = LoadUsdScene(scene_file, scale);
		}
		break;
		case SceneFormat::GLB:
		case SceneFormat::Unknown:
		default:
			AMBER_ERROR_LOG("Invalid or unsupported scene format: %s", scene_file);
		}

		if (scene && !environment_texture.empty())
		{
			scene->environment = std::make_unique<Image>(environment_texture.data());
		}
		return scene;
	}

}

