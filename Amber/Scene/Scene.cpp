#define CGLTF_IMPLEMENTATION
#include <unordered_map>
#include "Scene.h"
#include "Core/Logger.h"
#include "Math/MathCommon.h"
#include "pbrtParser/Scene.h"
#include "tinyobjloader/tiny_obj_loader.h"
#include "cgltf/cgltf.h"


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
				//AMBER_INFO("Loaded image texture: %s\n", t->fileName.c_str());
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
							//std::transform(
							//	mesh->vertex.begin(),
							//	mesh->vertex.end(),
							//	std::back_inserter(geom.vertices),
							//	[](pbrt::vec3f const& v) { return Vector3(v.x, v.y, v.z); });

							geom.indices.reserve(mesh->index.size());
							//std::transform(
							//	mesh->index.begin(),
							//	mesh->index.end(),
							//	std::back_inserter(geom.indices),
							//	[](pbrt::vec3i const& v) { return Vector3u(v.x, v.y, v.z); });

							geom.uvs.reserve(mesh->texcoord.size());
							//std::transform(mesh->texcoord.begin(),
							//	mesh->texcoord.end(),
							//	std::back_inserter(geom.uvs),
							//	[](pbrt::vec2f const& v) { return Vector2(v.x, v.y); });

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
						tinyobj::real_t vz = attrib.vertices[3 * uint64(idx.vertex_index) + 2] * scale * -1.0f;

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

			std::unordered_map<std::string, int32> texture_ids;
			for (auto const& m : materials)
			{
				Material material{};
				material.base_color = Vector3(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
				material.specular = Clamp(m.shininess / 500.f, 0.0f, 1.0f);
				material.roughness = Clamp(1.f - material.specular, 0.0f, 1.0f);
				material.specular_transmission = 0.0f;
				if (!m.diffuse_texname.empty()) 
				{
					if (!texture_ids.contains(m.diffuse_texname))
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
			cgltf_options options{};
			cgltf_data* gltf_data = nullptr;
			
			cgltf_result result = cgltf_parse_file(&options, scene_file.data(), &gltf_data);
			if (result != cgltf_result_success)
			{
				AMBER_WARN("GLTF - Failed to load '%s'", scene_file.data());
				return nullptr;
			}
			result = cgltf_load_buffers(&options, gltf_data, scene_file.data());
			if (result != cgltf_result_success)
			{
				AMBER_WARN("GLTF - Failed to load buffers '%s'", scene_file.data());
				return nullptr;
			}

			std::string gltf_base_dir = std::string(scene_file.substr(0, scene_file.rfind('/')));
			std::unique_ptr<Scene> gltf_scene = std::make_unique<Scene>();

			std::unordered_map<std::string, int32> texture_ids;
			gltf_scene->materials.reserve(gltf_data->materials_count);
			for (uint32 i = 0; i < gltf_data->materials_count; ++i)
			{
				cgltf_material const& gltf_material = gltf_data->materials[i];
				Material& material = gltf_scene->materials.emplace_back();

				cgltf_pbr_metallic_roughness pbr_metallic_roughness = gltf_material.pbr_metallic_roughness;
				material.base_color.x = (float)pbr_metallic_roughness.base_color_factor[0];
				material.base_color.y = (float)pbr_metallic_roughness.base_color_factor[1];
				material.base_color.z = (float)pbr_metallic_roughness.base_color_factor[2];
				material.metallic = (float)pbr_metallic_roughness.metallic_factor;
				material.roughness = (float)pbr_metallic_roughness.roughness_factor;
				material.emissive_color.x = (float)gltf_material.emissive_factor[0];
				material.emissive_color.y = (float)gltf_material.emissive_factor[1];
				material.emissive_color.z = (float)gltf_material.emissive_factor[2];
				material.alpha_cutoff = (float)gltf_material.alpha_cutoff;

				if (cgltf_texture* texture = pbr_metallic_roughness.base_color_texture.texture)
				{
					cgltf_image* image = texture->image;
					if (!texture_ids.contains(image->uri))
					{
						texture_ids[image->uri] = gltf_scene->textures.size();
						std::string texture_path = gltf_base_dir + "/" + image->uri;
						gltf_scene->textures.emplace_back(texture_path.c_str(), true);
					}
					const int32 id = texture_ids[image->uri];
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
					const int32 id = texture_ids[image->uri];
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
					const int32 id = texture_ids[image->uri];
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
					const int32 id = texture_ids[image->uri];
					material.emissive_tex_id = id;
				}

			}

			std::unordered_map<cgltf_mesh const*, std::vector<int32>> mesh_primitives_map; 
			int32 primitive_count = 0;

			for (uint32 i = 0; i < gltf_data->meshes_count; ++i)
			{
				cgltf_mesh const& gltf_mesh = gltf_data->meshes[i];
				std::vector<int32>& primitives = mesh_primitives_map[&gltf_mesh];

				Mesh& mesh = gltf_scene->meshes.emplace_back();
				for (uint32 j = 0; j < gltf_mesh.primitives_count; ++j)
				{
					auto const& gltf_primitive = gltf_mesh.primitives[j];
					AMBER_ASSERT(gltf_primitive.indices->count >= 0);

					Geometry& geometry = mesh.geometries.emplace_back();
					mesh.material_ids.push_back((int32)(gltf_primitive.material - gltf_data->materials));

					geometry.indices.reserve(gltf_primitive.indices->count / 3);

					uint32 triangle_cw[] = { 0, 1, 2 };
					uint32 triangle_ccw[] = { 0, 2, 1 };
					uint32* order = triangle_ccw;
					for (uint64 i = 0; i < gltf_primitive.indices->count; i += 3)
					{
						uint32 i0 = (uint32)cgltf_accessor_read_index(gltf_primitive.indices, i + order[0]);
						uint32 i1 = (uint32)cgltf_accessor_read_index(gltf_primitive.indices, i + order[1]);
						uint32 i2 = (uint32)cgltf_accessor_read_index(gltf_primitive.indices, i + order[2]);
						geometry.indices.emplace_back(i0, i1, i2);
					}

					for (uint32 k = 0; k < gltf_primitive.attributes_count; ++k)
					{
						cgltf_attribute const& gltf_attribute = gltf_primitive.attributes[k];
						std::string const& attr_name = gltf_attribute.name;

						auto ReadAttributeData = [&]<typename T>(std::vector<T>&stream, const char* stream_name)
						{
							if (!attr_name.compare(stream_name))
							{
								stream.resize(gltf_attribute.data->count);
								for (uint64 i = 0; i < gltf_attribute.data->count; ++i)
								{
									cgltf_accessor_read_float(gltf_attribute.data, i, &stream[i].x, sizeof(T) / sizeof(float));
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

			for (uint64 i = 0; i < gltf_data->nodes_count; ++i)
			{
				cgltf_node const& gltf_node = gltf_data->nodes[i];

				if (gltf_node.mesh)
				{
					Matrix local_to_world;
					cgltf_node_transform_world(&gltf_node, &local_to_world.m[0][0]);

					for (int32 primitive : mesh_primitives_map[gltf_node.mesh])
					{
						Instance& instance = gltf_scene->instances.emplace_back();
						instance.mesh_id = primitive;
						instance.transform = local_to_world * Matrix::CreateScale(scale, scale, -scale);
					}
				}
			}

			return gltf_scene;
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
			scene->environment = std::make_unique<Image>(environment_texture.data());
		}
		return scene;
	}

}

