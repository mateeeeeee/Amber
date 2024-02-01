#include <unordered_map>
#include <algorithm>
#include "Scene.h"

#include "Core/Logger.h"
#include "pbrtParser/Scene.h"

namespace lavender
{
	namespace
	{
		enum class SceneFormat : uint8
		{
			PBRT,
			PBF,
			Unknown
		};
		SceneFormat GetSceneFormat(std::string_view scene_file)
		{
			if (scene_file.ends_with(".pbrt")) return SceneFormat::PBRT;
			else if (scene_file.ends_with(".pbf")) return SceneFormat::PBF;
			else return SceneFormat::Unknown;
		}

		uint32 LoadPBRTMaterials(Scene& scene,
			pbrt::Material::SP const& mat,
			std::map<std::string, pbrt::Texture::SP> const& texture_overrides,
			std::string_view pbrt_base_dir,
			std::unordered_map<pbrt::Material::SP, uint64>& pbrt_materials,
			std::unordered_map<pbrt::Texture::SP, uint64>& pbrt_textures)
		{
			Material loaded_mat;
			const uint32 mat_id = scene.materials.size();
			pbrt_materials[mat] = mat_id;
			scene.materials.push_back(loaded_mat);
			return mat_id;
		}

		uint32 LoadPBRTTextures(
			pbrt::Texture::SP const& texture,
			std::string_view pbrt_base_dir,
			std::unordered_map<pbrt::Texture::SP, uint64>& pbrt_textures)
		{
			auto fnd = pbrt_textures.find(texture);
			if (fnd != pbrt_textures.end())
			{
				return fnd->second;
			}

			if (auto t = texture->as<pbrt::ImageTexture>())
			{
				std::string path = t->fileName;
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
				scene->bounding_box = BoundingBox(center, extents);
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
					LAV_WARN("Light source type %s not yet supported", pbrt_light->toString().c_str());
				}
			}

			std::unordered_map<pbrt::Material::SP, uint64>	pbrt_materials;
			std::unordered_map<pbrt::Texture::SP, uint64>	pbrt_textures;
			std::unordered_map<std::string, uint64>			pbrt_objects;
			for (auto const& instance : pbrt_world->instances)
			{
				uint64 primitive_id = -1;
				if (!pbrt_objects.contains(instance->object->name))
				{
					std::vector<uint32_t> material_ids;
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

					primitive_id = scene->primitives.size();
					scene->primitives.emplace_back(mesh_id, material_ids);

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
		}
	}

	std::unique_ptr<Scene> LoadScene(char const* _scene_file)
	{
		std::string_view scene_file(_scene_file);
		SceneFormat scene_format = GetSceneFormat(scene_file);
		switch (scene_format)
		{
		case SceneFormat::PBRT:
		{
			std::shared_ptr<pbrt::Scene> pbrt_scene = pbrt::importPBRT(_scene_file);
			return ConvertPBRTScene(pbrt_scene, scene_file);
		}
		break;
		case SceneFormat::PBF:
		{
			std::shared_ptr<pbrt::Scene> pbrt_scene = pbrt::Scene::loadFrom(_scene_file);
			return ConvertPBRTScene(pbrt_scene, scene_file);
		}
		break;
		case SceneFormat::Unknown:
		default:
			LAV_ERROR("Invalid scene format: %s", scene_file);
		}
		return nullptr;
	}

}

