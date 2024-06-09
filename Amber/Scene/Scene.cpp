#include <unordered_map>
#include <algorithm>
#include "Scene.h"
#include "Core/Logger.h"
#include "pbrtParser/Scene.h"
#include "tinyobjloader/tiny_obj_loader.h"


namespace amber
{
	enum class SceneFormat : uint8
	{
		OBJ,
		GLTF,
		GLB,
		PBRT,
		PBF,
		Unknown
	};

	SceneFormat GetSceneFormat(std::string_view scene_file)
	{
		if (scene_file.ends_with(".pbrt")) return SceneFormat::PBRT;
		else if (scene_file.ends_with(".pbf")) return SceneFormat::PBF;
		else if (scene_file.ends_with(".obj")) return SceneFormat::OBJ;
		else if (scene_file.ends_with(".gltf")) return SceneFormat::GLTF;
		else if (scene_file.ends_with(".glb")) return SceneFormat::GLB;
		else return SceneFormat::Unknown;
	}
	
	namespace 
	{
		int32 LoadPBRTTexture(
			Scene& scene,
			pbrt::Texture::SP const& texture,
			std::string_view pbrt_base_dir,
			std::unordered_map<pbrt::Texture::SP, uint32>& pbrt_textures)
		{
			auto fnd = pbrt_textures.find(texture);
			if (fnd != pbrt_textures.end())
			{
				return fnd->second;
			}

			if (auto t = texture->as<pbrt::ImageTexture>())
			{
				std::string path = std::string(pbrt_base_dir) + t->fileName;
				const uint32 id = scene.textures.size();
				pbrt_textures[texture] = id;
				//scene.textures.emplace_back(path.c_str());
				//LAV_INFO("Loaded image texture: %s\n", t->fileName.c_str());
				return id;
			}
			else if (auto t = texture->as<pbrt::ConstantTexture>())
			{
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

		uint32 LoadPBRTMaterials(Scene& scene,
			pbrt::Material::SP const& mat,
			std::map<std::string, pbrt::Texture::SP> const& texture_overrides,
			std::string_view pbrt_base_dir,
			std::unordered_map<pbrt::Material::SP, uint32>& pbrt_materials,
			std::unordered_map<pbrt::Texture::SP, uint32>& pbrt_textures)
		{
			if (pbrt_materials.contains(mat))
			{
				return pbrt_materials[mat];
			}

			Material loaded_mat{};
			if (auto m = mat->as<pbrt::DisneyMaterial>())
			{
				loaded_mat.anisotropy = m->anisotropic;
				loaded_mat.clearcoat = m->clearCoat;
				loaded_mat.clearcoat_gloss = m->clearCoatGloss;
				loaded_mat.base_color = Vector3(m->color.x, m->color.y, m->color.z);
				loaded_mat.ior = m->eta;
				loaded_mat.metallic = m->metallic;
				loaded_mat.roughness = m->roughness;
				loaded_mat.sheen = m->sheen;
				loaded_mat.sheen_tint = m->sheenTint;
				loaded_mat.specular_tint = m->specularTint;
				loaded_mat.specular = 0.0f;
			}
			else if (auto m = mat->as<pbrt::PlasticMaterial>())
			{
				loaded_mat.base_color = Vector3(m->kd.x, m->kd.y, m->kd.z);
				if (m->map_kd) 
				{
					if (auto const_tex = m->map_kd->as<pbrt::ConstantTexture>())
					{
						loaded_mat.base_color = Vector3(const_tex->value.x, const_tex->value.y, const_tex->value.z);
					}
					else 
					{
						int32 tex_id = LoadPBRTTexture(scene, m->map_kd, pbrt_base_dir, pbrt_textures);
						loaded_mat.diffuse_tex_id = tex_id;
					}
				}
				Vector3 ks(m->ks.x, m->ks.y, m->ks.z);
				loaded_mat.specular = ks.x * 0.2126f +  0.7152f * ks.y + 0.0722f * ks.z;
				loaded_mat.roughness = m->roughness;
			}
			else if (auto m = mat->as<pbrt::MatteMaterial>())
			{
				loaded_mat.base_color = Vector3(m->kd.x, m->kd.y, m->kd.z);
				if (m->map_kd) 
				{
					if (auto const_tex = m->map_kd->as<pbrt::ConstantTexture>())
					{
						loaded_mat.base_color = Vector3(const_tex->value.x, const_tex->value.y, const_tex->value.z);
					}
					else 
					{
						int32 tex_id = LoadPBRTTexture(scene, m->map_kd, pbrt_base_dir, pbrt_textures);
						loaded_mat.diffuse_tex_id = tex_id;
					}
				}
			}
			else if (auto m = mat->as<pbrt::SubstrateMaterial>())
			{
				loaded_mat.base_color = Vector3(m->kd.x, m->kd.y, m->kd.z);
				if (m->map_kd) 
				{
					if (auto const_tex = m->map_kd->as<pbrt::ConstantTexture>())
					{
						loaded_mat.base_color = Vector3(const_tex->value.x, const_tex->value.y, const_tex->value.z);
					}
					else 
					{
						int32 tex_id = LoadPBRTTexture(scene, m->map_kd, pbrt_base_dir, pbrt_textures);
						loaded_mat.diffuse_tex_id = tex_id;
					}
				}
				Vector3 ks(m->ks.x, m->ks.y, m->ks.z);
				loaded_mat.specular = ks.x * 0.2126f + 0.7152f * ks.y + 0.0722f * ks.z;
				loaded_mat.roughness = 1.0f;
				loaded_mat.clearcoat = 1.0f;
				loaded_mat.clearcoat_gloss = loaded_mat.specular;
			}

			uint32 mat_id = scene.materials.size();
			pbrt_materials[mat] = mat_id;
			scene.materials.push_back(loaded_mat);
			return mat_id;
		}

		std::unique_ptr<Scene> ConvertPBRTScene(std::shared_ptr<pbrt::Scene> const& pbrt_scene, std::string_view scene_file)
		{
			pbrt_scene->makeSingleLevel();
			std::unique_ptr<Scene> scene = std::make_unique<Scene>();
			auto const& pbrt_world = pbrt_scene->world;
			if (pbrt_world->haveComputedBounds)
			{
				pbrt::box3f pbrt_bounds = pbrt_world->getBounds();
				pbrt::vec3f pbrt_center = (pbrt_bounds.lower + pbrt_bounds.upper) * 0.5f;
				pbrt::vec3f pbrt_extents = (pbrt_bounds.upper - pbrt_bounds.lower) * 0.5f;
				Vector3 center(&pbrt_center.x);
				Vector3 extents(&pbrt_extents.x);
			}
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
					AMBER_WARN("Light source type %s not yet supported", pbrt_light->toString().c_str());
				}
			}

			std::unordered_map<pbrt::Material::SP, uint32>	pbrt_materials;
			std::unordered_map<pbrt::Texture::SP, uint32>	pbrt_textures;
			std::unordered_map<std::string, uint32>			pbrt_objects;
			for (auto const& instance : pbrt_world->instances)
			{
				uint64 primitive_id = -1;
				if (!pbrt_objects.contains(instance->object->name))
				{
					std::vector<uint32> material_ids;
					std::vector<Geometry> geometries;
					for (const auto& shape : instance->object->shapes)
					{
						if (pbrt::TriangleMesh::SP mesh = shape->as<pbrt::TriangleMesh>())
						{
							uint32 material_id = -1;
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
							std::transform(
								mesh->vertex.begin(),
								mesh->vertex.end(),
								std::back_inserter(geom.vertices),
								[](pbrt::vec3f const& v) { return Vector3(v.x, v.y, v.z); });

							geom.indices.reserve(mesh->index.size());
							std::transform(
								mesh->index.begin(),
								mesh->index.end(),
								std::back_inserter(geom.indices),
								[](pbrt::vec3i const& v) { return Vector3u(v.x, v.y, v.z); });

							geom.uvs.reserve(mesh->texcoord.size());
							std::transform(mesh->texcoord.begin(),
								mesh->texcoord.end(),
								std::back_inserter(geom.uvs),
								[](pbrt::vec2f const& v) { return Vector2(v.x, v.y); });

							geometries.push_back(geom);
						}
						if (pbrt::Sphere::SP mesh = shape->as<pbrt::Sphere>())
						{

						}
					}

					uint64 const mesh_id = scene->meshes.size();
					scene->meshes.emplace_back(geometries);

					pbrt_objects[instance->object->name] = mesh_id;
				}
				else
				{
					auto fnd = pbrt_objects.find(instance->object->name);
					primitive_id = fnd->second;
				}

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

		std::unique_ptr<Scene> LoadObjScene(std::string_view scene_file, float scale)
		{
			tinyobj::ObjReaderConfig reader_config{};
			tinyobj::ObjReader reader;
			if (!reader.ParseFromFile(std::string(scene_file), reader_config))
			{
				if (!reader.Error().empty())
				{
					AMBER_ERROR("TinyOBJ error: %s", reader.Error().c_str());
				}
				return nullptr;
			}
			if (!reader.Warning().empty())
			{
				AMBER_WARN("TinyOBJ warning: %s", reader.Warning().c_str());
			}

			std::string obj_base_dir = std::string(scene_file.substr(0, scene_file.rfind('/')));
			tinyobj::attrib_t const& attrib = reader.GetAttrib();
			std::vector<tinyobj::shape_t> const& shapes = reader.GetShapes();
			std::vector<tinyobj::material_t> const& materials = reader.GetMaterials();

			std::unique_ptr<Scene> obj_scene = std::make_unique<Scene>();

			Mesh mesh;
			std::vector<uint32> material_ids;
			for (uint64 s = 0; s < shapes.size(); s++)
			{
				tinyobj::mesh_t const& obj_mesh = shapes[s].mesh;
				if(obj_mesh.material_ids[0] >= 0) material_ids.push_back(obj_mesh.material_ids[0]);
				
				Geometry geometry{};
				uint32 index_offset = 0;
				for (uint64 f = 0; f < obj_mesh.num_face_vertices.size(); ++f) 
				{
					AMBER_ASSERT(obj_mesh.num_face_vertices[f] == 3);
					for (uint64 v = 0; v < 3; v++)
					{
						tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
						tinyobj::real_t vx = attrib.vertices[3 * uint64(idx.vertex_index) + 0] * scale;
						tinyobj::real_t vy = attrib.vertices[3 * uint64(idx.vertex_index) + 1] * scale;
						tinyobj::real_t vz = attrib.vertices[3 * uint64(idx.vertex_index) + 2] * scale;

						geometry.vertices.emplace_back(vx, vy, vz);

						if (idx.normal_index >= 0)
						{
							tinyobj::real_t nx = attrib.normals[3 * uint64(idx.normal_index) + 0];
							tinyobj::real_t ny = attrib.normals[3 * uint64(idx.normal_index) + 1];
							tinyobj::real_t nz = attrib.normals[3 * uint64(idx.normal_index) + 2];
							geometry.normals.emplace_back(nx, ny, nz);
						}

						if (idx.texcoord_index >= 0)
						{
							tinyobj::real_t tx = attrib.texcoords[2 * uint64(idx.texcoord_index) + 0];
							tinyobj::real_t ty = attrib.texcoords[2 * uint64(idx.texcoord_index) + 1];

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
			
			auto clamp = []<typename T>(T v, T min, T max)
			{
				return v < min ? min : (v > max ? max : v);
			};

			std::unordered_map<std::string, int32> texture_ids;
			for (auto const& m : materials)
			{
				Material material{};
				material.base_color = Vector3(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
				material.specular = clamp(m.shininess / 500.f, 0.0f, 1.0f);
				material.roughness = clamp(1.f - material.specular, 0.0f, 1.0f);
				material.specular_transmission = 0.0f;
				if (!m.diffuse_texname.empty()) 
				{
					if (texture_ids.find(m.diffuse_texname) == texture_ids.end())
					{
						texture_ids[m.diffuse_texname] = obj_scene->textures.size();
						std::string texture_path = obj_base_dir + "/" + m.diffuse_texname;
						obj_scene->textures.emplace_back(texture_path.c_str(), true);
					}
					const int32 id = texture_ids[m.diffuse_texname];
					material.diffuse_tex_id = id;
				}
				obj_scene->materials.push_back(material);
			}

			return obj_scene;
		}

		std::unique_ptr<Scene> LoadGltfScene(std::string_view scene_file, float scale)
		{
			AMBER_ASSERT(false);
			return nullptr;
		}
	}

	std::unique_ptr<Scene> LoadScene(char const* _scene_file, char const* _environment_texture, float scale)
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
		case SceneFormat::GLB:
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
		case SceneFormat::Unknown:
		default:
			AMBER_ERROR("Invalid scene format: %s", scene_file);
		}

		if (scene && !environment_texture.empty())
		{
			//scene->environment = std::make_unique<Image>(environment_texture.data());
		}
		return scene;
	}

}

