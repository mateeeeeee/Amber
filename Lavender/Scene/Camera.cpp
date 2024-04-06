#include "Camera.h"
#include "Core/Input.h"

namespace lavender
{

	void Camera::Tick(float dt)
	{
		Input& input = g_Input;
		if (input.GetKey(KeyCode::Space)) return;

		float speed_factor = 1.0f;
		if (input.GetKey(KeyCode::ShiftLeft)) speed_factor *= 5.0f;
		if (input.GetKey(KeyCode::CtrlLeft))  speed_factor *= 0.2f;

		Vector3 u, v, w;
		GetFrame(u, v, w);

		if (input.GetKey(KeyCode::W))
		{
			eye += speed_factor * dt * w;
		}
		if (input.GetKey(KeyCode::S))
		{
			eye -= speed_factor * dt * w;
		}
		if (input.GetKey(KeyCode::A))
		{
			eye -= speed_factor * dt * u;
		}
		if (input.GetKey(KeyCode::D))
		{
			eye += speed_factor * dt * u;
		}
		if (input.GetKey(KeyCode::Q))
		{
			eye += speed_factor * dt * v;
		}
		if (input.GetKey(KeyCode::E))
		{
			eye -= speed_factor * dt * v;
		}
		if (input.GetKey(KeyCode::MouseRight))
		{
			float dx = input.GetMouseDeltaX();
			float dy = input.GetMouseDeltaY();
			//Pitch((int64)dy);
			//Yaw((int64)dx);
		}
	}


	void Camera::GetFrame(Vector3& U, Vector3& V, Vector3& W) const
	{
		W = look_dir;
		U = W.Cross(up); U.Normalize();
		V = U.Cross(W); V.Normalize();
	}

}


