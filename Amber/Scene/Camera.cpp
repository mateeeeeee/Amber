#include "Camera.h"
#include "Core/Input.h"

namespace amber
{
	static constexpr float speed = 0.1f;

	void Camera::Initialize(Vector3 const& eye, Vector3 const& look_at)
	{
		position = eye;
		Vector3 look_vector = look_at - position;
		look_vector.Normalize();

		//Vector3 up_vector = Vector3::Up - Vector3::Up.Dot(look_vector) * look_vector;
		//up_vector.Normalize();
		orientation = Quaternion::LookRotation(look_vector, Vector3::Up);
		fovy = 45.0f;
		aspect_ratio = 1.0f;
		changed = false;
	}

	void Camera::Update(float dt)
	{
		changed = false;
		if (!enabled) return;
		Input& input = g_Input;
		if (input.GetKey(KeyCode::Space)) return;

		if (g_Input.GetKey(KeyCode::MouseRight))
		{
			float dx = g_Input.GetMouseDeltaX();
			float dy = g_Input.GetMouseDeltaY();
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
		velocity = Vector3::SmoothStep(velocity, movement, 0.35f);

		if (velocity.LengthSquared() > 1e-4)
		{
			float speed_factor = 1.0f;
			if (input.GetKey(KeyCode::ShiftLeft)) speed_factor *= 5.0f;
			if (input.GetKey(KeyCode::CtrlLeft))  speed_factor *= 0.2f;
			position += velocity * dt * speed_factor * 5.0f;
			changed = true;
		}
	}

	Vector3 Camera::GetLookDir() const
	{
		return Vector3::Transform(Vector3::Forward, orientation);
	}

	void Camera::SetLookDir(Vector3 look_dir)
	{
		look_dir.Normalize();
		orientation = Quaternion::LookRotation(look_dir, Vector3::Up);
	}

	void Camera::GetFrame(Vector3& U, Vector3& V, Vector3& W) const
	{
		U = Vector3::Transform(Vector3::Right, orientation);
		V = Vector3::Transform(Vector3::Up, orientation);
		W = Vector3::Transform(Vector3::Forward, orientation);
	}
}


