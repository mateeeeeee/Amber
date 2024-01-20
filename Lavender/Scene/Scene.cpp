#include "Scene.h"
#include "Camera.h"
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
					pbrt::vec3f pbrt_dir = pbrt_distant_light->to - pbrt_distant_light->from;
					pbrt::vec3f pbrt_spectrum = pbrt_distant_light->L * pbrt_distant_light->scale;
					pbrt::affine3f pbrt_transform = pbrt_distant_light->transform;
					
					Vector3 dir(&pbrt_dir.x); //multiply by transform?
					Vector3 spectrum(&pbrt_spectrum.x);
					Light light = MakeLight<LightType::Directional>(dir);
					light.color = spectrum;
					scene->lights.push_back(light);
				}
				else if (auto pbrt_point_light = pbrt_light->as<pbrt::PointLightSource>())
				{
					pbrt::vec3f pbrt_pos = pbrt_point_light->from;
					pbrt::vec3f pbrt_spectrum = pbrt_point_light->I * pbrt_point_light->scale;

					Vector3 pos(&pbrt_pos.x);
					Vector3 spectrum(&pbrt_spectrum.x);
					Light light = MakeLight<LightType::Point>(pos);
					scene->lights.push_back(light);
					light.color = spectrum;
					scene->lights.push_back(light);
				}
				else
				{
					LAV_WARN("Light source type %s not yet supported", pbrt_light->toString().c_str());
				}
			}

			for (auto const& pbrt_shape : pbrt_world->shapes)
			{

			}

			for (auto const& pbrt_instance : pbrt_world->instances)
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

