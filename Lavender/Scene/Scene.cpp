#include <unordered_map>
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

		Matrix ConvertPBRTTransform(pbrt::affine3f const& pbrt_transform)
		{
			pbrt::vec3f const& vx = pbrt_transform.l.vx;
			pbrt::vec3f const& vy = pbrt_transform.l.vy;
			pbrt::vec3f const& vz = pbrt_transform.l.vz;
			pbrt::vec3f const& p  = pbrt_transform.p;

			Vector4 v0(vx.x, vx.y, vx.z, 0.0f);
			Vector4 v1(vy.x, vy.y, vy.z, 0.0f);
			Vector4 v2(vz.x, vz.y, vz.z, 0.0f);
			Vector4 v3(p.x, p.y, p.z, 1.0f);
			return Matrix(v0, v1, v2, v3);
		}

		std::unique_ptr<Scene> ConvertPBRTScene(std::shared_ptr<pbrt::Scene> const& pbrt_scene)
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

			}

			return scene;
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
			return ConvertPBRTScene(pbrt_scene);
		}
		break;
		case SceneFormat::PBF: 
		{
			std::shared_ptr<pbrt::Scene> pbrt_scene = pbrt::Scene::loadFrom(_scene_file); 
			return ConvertPBRTScene(pbrt_scene);
		}
		break;
		case SceneFormat::Unknown: 
		default:
			LAV_ERROR("Invalid scene format: %s", scene_file);
		}
		return nullptr;
	}

}

