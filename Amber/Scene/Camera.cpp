#include "Camera.h"
#include "Core/Input.h"

namespace amber
{

	void Camera::Tick(float dt)
	{
		Input& input = g_Input;
		if (input.GetKey(KeyCode::Space)) return;

		float speed_factor = 1.0f;
		if (input.GetKey(KeyCode::ShiftLeft)) speed_factor *= 5.0f;
		if (input.GetKey(KeyCode::CtrlLeft))  speed_factor *= 0.2f;

		if (input.GetKey(KeyCode::W))
		{
			eye += speed_factor * dt * look_dir;
		}
		if (input.GetKey(KeyCode::S))
		{
			eye -= speed_factor * dt * look_dir;
		}
		if (input.GetKey(KeyCode::A))
		{
			eye -= speed_factor * dt * right;
		}
		if (input.GetKey(KeyCode::D))
		{
			eye += speed_factor * dt * right;
		}
		if (input.GetKey(KeyCode::Q))
		{
			eye += speed_factor * dt * up;
		}
		if (input.GetKey(KeyCode::E))
		{
			eye -= speed_factor * dt * up;
		}
		if (input.GetKey(KeyCode::MouseRight))
		{
			float dx = -input.GetMouseDeltaX();
			float dy = -input.GetMouseDeltaY();

			Matrix R = Matrix::CreateFromAxisAngle(right, 0.2f * DirectX::XMConvertToRadians(dy));
			up = Vector3::TransformNormal(up, R);
			look_dir = Vector3::TransformNormal(look_dir, R);

			R = Matrix::CreateRotationY(0.2f * DirectX::XMConvertToRadians(dx));
			right = Vector3::TransformNormal(right, R);
			up = Vector3::TransformNormal(up, R);
			look_dir = Vector3::TransformNormal(look_dir, R);
		}

		UpdateFrame();
	}


	void Camera::GetFrame(Vector3& U, Vector3& V, Vector3& W) const
	{
		U = right;
		V = up;
		W = look_dir;
	}

	void Camera::UpdateFrame()
	{
		look_dir.Normalize();
		up = look_dir.Cross(right);
		up.Normalize();
		right = up.Cross(look_dir);
	}

}


