#include "Camera.h"
#include "Core/Input.h"

namespace amber
{
	static constexpr Float speed = 0.1f;

	void Camera::Initialize(Vector3 const& eye, Vector3 const& look_at)
	{
		position = eye;
		Vector3 look_vector = look_at - position;
		look_vector.Normalize();

		Float yaw = std::atan2(look_vector.x, look_vector.z);
		Float pitch = std::asin(-look_vector.y);
		Quaternion pitch_quat = Quaternion::CreateFromYawPitchRoll(0, pitch, 0);
		Quaternion yaw_quat = Quaternion::CreateFromYawPitchRoll(yaw, 0, 0);
		orientation = pitch_quat * orientation * yaw_quat;
		fovy = 45.0f;
		aspect_ratio = 1.0f;
		changed = false;
	}

	void Camera::Update(Float dt)
	{
		changed = false;
		if (!enabled || g_Input.GetKey(KeyCode::Space))
		{
			return;
		}

		if (g_Input.GetKey(KeyCode::MouseRight))
		{
			Float dx = g_Input.GetMouseDeltaX();
			Float dy = g_Input.GetMouseDeltaY();
			Quaternion yaw_quaternion = Quaternion::CreateFromYawPitchRoll(0, dy * dt * 0.25f, 0);
			Quaternion pitch_quaternion = Quaternion::CreateFromYawPitchRoll(dx * dt * 0.25f, 0, 0);
			orientation = yaw_quaternion * orientation * pitch_quaternion;
			changed = true;
		}

		Vector3 movement{};
		if (g_Input.GetKey(KeyCode::W)) movement.z += 1.0f;
		if (g_Input.GetKey(KeyCode::S)) movement.z -= 1.0f;
		if (g_Input.GetKey(KeyCode::D)) movement.x += 1.0f;
		if (g_Input.GetKey(KeyCode::A)) movement.x -= 1.0f;
		if (g_Input.GetKey(KeyCode::Q)) movement.y -= 1.0f;
		if (g_Input.GetKey(KeyCode::E)) movement.y += 1.0f;
		movement = Vector3::Transform(movement, orientation);
		velocity = Vector3::SmoothStep(velocity, movement, 0.25f);

		if (velocity.LengthSquared() > 1e-4)
		{
			Float speed_factor = 1.0f;
			if (g_Input.GetKey(KeyCode::ShiftLeft)) speed_factor *= 5.0f;
			if (g_Input.GetKey(KeyCode::CtrlLeft))  speed_factor *= 0.2f;
			position += velocity * dt * speed_factor * 5.0f;
			changed = true;
		}
	}

	Vector3 Camera::GetLookDir() const
	{
		return Vector3::Transform(Vector3::Forward, orientation);
	}

	void Camera::SetLookDir(Vector3 look_vector)
	{
		look_vector.Normalize();
		Float yaw = std::atan2(look_vector.x, look_vector.z);
		Float pitch = std::asin(-look_vector.y);
		Quaternion pitch_quat = Quaternion::CreateFromYawPitchRoll(0, pitch, 0);
		Quaternion yaw_quat = Quaternion::CreateFromYawPitchRoll(yaw, 0, 0);
		orientation = pitch_quat * orientation * yaw_quat;
	}

	void Camera::GetFrame(Vector3& U, Vector3& V, Vector3& W) const
	{
		U = Vector3::Transform(Vector3::Right, orientation);
		V = Vector3::Transform(Vector3::Up, orientation);
		W = Vector3::Transform(Vector3::Forward, orientation);
	}
}


