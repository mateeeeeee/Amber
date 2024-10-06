#pragma once

namespace amber
{
	class Camera
	{
	public:
		void Initialize(Vector3 const& position, Vector3 const& lookat);

		void Update(float dt);
		void Enable(bool _enabled) { enabled = _enabled; }
		bool IsChanged() const { return changed; }

		Vector3 const& GetPosition() const { return position; }
		void SetPosition(Vector3 const& val) { position = val; }
		Vector3 GetLookDir() const;
		void SetLookDir(Vector3 val);

		float GetFovY() const { return fovy; }
		void  SetFovY(float val) { fovy = val; }
		float GetAspectRatio() const { return aspect_ratio; }
		void  SetAspectRatio(float val) { aspect_ratio = val; }

		void GetFrame(Vector3& U, Vector3& V, Vector3& W) const;

	private:
		Vector3		position;
		Vector3     velocity;
		Quaternion  orientation;

		float fovy;
		float aspect_ratio;
		bool changed;
		bool enabled;
	};
}