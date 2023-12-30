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
			else if (scene_file.ends_with(".pbrt")) return SceneFormat::PBF;
			else return SceneFormat::Unknown;
		}
		std::unique_ptr<Scene> ConvertPBRTScene(std::shared_ptr<pbrt::Scene> const& pbrt_scene)
		{
			return std::make_unique<Scene>();
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
			LAV_ERROR("Invalid scene format: {}", scene_file);
		}
		return nullptr;
	}

}

