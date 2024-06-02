#pragma once

namespace amber
{
	class Camera 
	{
		static constexpr float speed = 0.1f;
	public:
		Camera()  : eye(Vector3(0.0f, 0.0f, 1.0f)), look_dir(Vector3(0.0f, 0.0f, -1.0f))
			, up(Vector3(0.0f, 1.0f, 0.0f)), right(Vector3(1.0f, 0.0f, 0.0f))
			, fovy(35.0f), aspect_ratio(1.0f)
		{
		}

		Vector3 const& GetEye() const { return eye; }
		void SetEye(Vector3 const& val) { eye = val; }

		void Tick(float dt);

		void SetLookat(Vector3 const& val) 
		{ 
			look_dir = val - eye;
			look_dir.Normalize();
		}
		Vector3 GetLookDir() const
		{
			return look_dir;
		}
		void SetLookDir(Vector3 const& dir)
		{
			look_dir = dir;
			look_dir.Normalize();
		}

		Vector3 const& GetUp() const { return up; }
		void SetUp(Vector3 const& val) { up = val; }
		float GetFovY() const { return fovy; }
		void SetFovY(float val) { fovy = val; }
		float GetAspectRatio() const { return aspect_ratio; }
		void  SetAspectRatio(float val) { aspect_ratio = val; }

		void GetFrame(Vector3& U, Vector3& V, Vector3& W) const;

	private:
		Vector3 eye;
		Vector3 look_dir;
		Vector3 up;
		Vector3 right;
		float fovy;
		float aspect_ratio;

	private:
		void UpdateFrame();
	};
}