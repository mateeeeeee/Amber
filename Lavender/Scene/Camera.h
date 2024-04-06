#pragma once

namespace lavender
{
	class Camera 
	{
		static constexpr float speed = 0.1f;
	public:
		Camera() : eye(Vector3(1.0f, 1.0f, 1.0f)), lookat(Vector3(0.0f, 0.0f, 0.0f)), up(Vector3(0.0f, 1.0f, 0.0f)), fovy(35.0f), aspect_ratio(1.0f)
		{
		}

		Camera(Vector3 const& eye, Vector3 const& lookat, Vector3 const& up, float fovY, float aspectRatio)
			: eye(eye), lookat(lookat), up(up), fovy(fovY), aspect_ratio(aspectRatio)
		{
		}

		Vector3 GetDirection() const 
		{ 
			Vector3 dir = lookat - eye;
			Vector3 normalized_dir;
			dir.Normalize(normalized_dir);
			return normalized_dir;
		}
		void SetDirection(Vector3 const& dir) 
		{ 
			lookat = eye + dir * dir.Length();
		}

		Vector3 const& GetEye() const { return eye; }
		void SetEye(Vector3 const& val) { eye = val; }

		void Tick(float dt);

		Vector3 const& GetLookat() const { return lookat; }
		void SetLookat(Vector3 const& val) { lookat = val; }

		Vector3 const& GetUp() const { return up; }
		void SetUp(Vector3 const& val) { up = val; }
		float GetFovY() const { return fovy; }
		void SetFovY(float val) { fovy = val; }
		float GetAspectRatio() const { return aspect_ratio; }
		void  SetAspectRatio(float val) { aspect_ratio = val; }

		void GetFrame(Vector3& U, Vector3& V, Vector3& W) const;

	private:
		Vector3 eye;

		Vector3 lookat;
		Vector3 up;
		float fovy;
		float aspect_ratio;
	};
}