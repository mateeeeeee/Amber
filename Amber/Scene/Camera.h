#pragma once

namespace amber
{
	class Camera
	{
	public:
		void Initialize(Vector3 const& position, Vector3 const& lookat);

		void Update(Float dt);
		void Enable(Bool _enabled) { enabled = _enabled; }
		Bool IsChanged() const { return changed; }

		Vector3 const& GetPosition() const { return position; }
		void SetPosition(Vector3 const& val) { position = val; }
		Vector3 GetLookDir() const;
		void SetLookDir(Vector3 val);

		Float GetFovY() const { return fovy; }
		void  SetFovY(Float val) { fovy = val; }
		Float GetAspectRatio() const { return aspect_ratio; }
		void  SetAspectRatio(Float val) { aspect_ratio = val; }

		void GetFrame(Vector3& U, Vector3& V, Vector3& W) const;

	private:
		Vector3		position;
		Vector3     velocity;
		Quaternion  orientation;

		Float fovy;
		Float aspect_ratio;
		Bool changed;
		Bool enabled;
	};
}