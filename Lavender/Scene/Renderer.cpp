#include <fstream>
#include "Renderer.h"
#include "Core/Logger.h"
#include "Core/Paths.h"
#include "Utilities/JsonUtil.h"

namespace lavender
{
	Renderer::Renderer(char const* config_file)
	{
		json camera, scene;
		try
		{
			JsonParams scene_params = json::parse(std::ifstream(paths::ConfigDir() + config_file));
			camera = scene_params.FindJson("camera");
			scene = scene_params.FindJson("scene");
		}
		catch (json::parse_error const& e)
		{
			LAVENDER_ERROR("JSON Parse error: %s! ", e.what());
			return;
		}
		JsonParams scene_params(scene);
		JsonParams camera_params(camera);
		//scene_params.Find<std::string>("")
		//
		//config.camera_params.near_plane = camera_params.FindOr<float>("near", 1.0f);
		//config.camera_params.far_plane = camera_params.FindOr<float>("far", 3000.0f);
		//config.camera_params.fov = XMConvertToRadians(camera_params.FindOr<float>("fov", 90.0f));
		//config.camera_params.sensitivity = camera_params.FindOr<float>("sensitivity", 0.3f);
		//config.camera_params.speed = camera_params.FindOr<float>("speed", 25.0f);
		//
		//float position[3] = { 0.0f, 0.0f, 0.0f };
		//camera_params.FindArray("position", position);
		//config.camera_params.position = Vector3(position);
		//
		//float look_at[3] = { 0.0f, 0.0f, 10.0f };
		//camera_params.FindArray("look_at", look_at);
		//config.camera_params.look_at = Vector3(look_at);
		//
	}

	void Renderer::Update(float dt)
	{

	}

	void Renderer::Render()
	{

	}

}

