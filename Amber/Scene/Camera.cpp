#include "Camera.h"
#include "Core/Input.h"

namespace amber
{
	static constexpr float speed = 0.1f;

	void Camera::Update(float dt)
	{
		changed = false;
		Input& input = g_Input;
		if (input.GetKey(KeyCode::Space)) return;

		float speed_factor = 1.0f;
		if (input.GetKey(KeyCode::ShiftLeft)) speed_factor *= 5.0f;
		if (input.GetKey(KeyCode::CtrlLeft))  speed_factor *= 0.2f;

		if (input.GetKey(KeyCode::W))
		{
			eye += speed_factor * dt * look_dir;
			changed = true;
		}
		if (input.GetKey(KeyCode::S))
		{
			eye -= speed_factor * dt * look_dir;
			changed = true;
		}
		if (input.GetKey(KeyCode::A))
		{
			eye -= speed_factor * dt * right;
			changed = true;
		}
		if (input.GetKey(KeyCode::D))
		{
			eye += speed_factor * dt * right;
			changed = true;
		}
		if (input.GetKey(KeyCode::Q))
		{
			eye += speed_factor * dt * up;
			changed = true;
		}
		if (input.GetKey(KeyCode::E))
		{
			eye -= speed_factor * dt * up;
			changed = true;
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
			changed = true;
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


