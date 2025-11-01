#pragma once
#include <cmath>

namespace amber
{
	struct Vector2
	{
		Float x, y;

		Vector2() : x(0.0f), y(0.0f) {}
		Vector2(Float x, Float y) : x(x), y(y) {}

		Vector2 operator+(Vector2 const& v) const { return Vector2(x + v.x, y + v.y); }
		Vector2 operator-(Vector2 const& v) const { return Vector2(x - v.x, y - v.y); }
		Vector2 operator*(Float s) const { return Vector2(x * s, y * s); }
		Vector2 operator/(Float s) const { return Vector2(x / s, y / s); }

		Float Length() const { return std::sqrt(x * x + y * y); }
		Vector2 Normalized() const { Float len = Length(); return len > 0 ? *this / len : Vector2(); }
	};

	struct Vector3
	{
		Float x, y, z;

		Vector3() : x(0.0f), y(0.0f), z(0.0f) {}
		Vector3(Float x, Float y, Float z) : x(x), y(y), z(z) {}
		Vector3(Float* arr) : x(arr[0]), y(arr[1]), z(arr[2]) {}

		Vector3 operator+(Vector3 const& v) const { return Vector3(x + v.x, y + v.y, z + v.z); }
		Vector3 operator-(Vector3 const& v) const { return Vector3(x - v.x, y - v.y, z - v.z); }
		Vector3 operator*(Float s) const { return Vector3(x * s, y * s, z * s); }
		Vector3 operator/(Float s) const { return Vector3(x / s, y / s, z / s); }
		Vector3 operator-() const { return Vector3(-x, -y, -z); }

		Vector3& operator+=(Vector3 const& v) { x += v.x; y += v.y; z += v.z; return *this; }
		Vector3& operator-=(Vector3 const& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
		Vector3& operator*=(Float s) { x *= s; y *= s; z *= s; return *this; }
		Vector3& operator/=(Float s) { x /= s; y /= s; z /= s; return *this; }

		Float Length() const { return std::sqrt(x * x + y * y + z * z); }
		Float LengthSquared() const { return x * x + y * y + z * z; }
		Vector3 Normalized() const { Float len = Length(); return len > 0 ? *this / len : Vector3(); }
		void Normalize() { Float len = Length(); if (len > 0) { x /= len; y /= len; z /= len; } }

		Float Dot(Vector3 const& v) const { return x * v.x + y * v.y + z * v.z; }
		static Float Dot(Vector3 const& a, Vector3 const& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
		static Vector3 Cross(Vector3 const& a, Vector3 const& b)
		{
			return Vector3(
				a.y * b.z - a.z * b.y,
				a.z * b.x - a.x * b.z,
				a.x * b.y - a.y * b.x
			);
		}

		// Transform vector by quaternion
		static Vector3 Transform(Vector3 const& v, struct Quaternion const& q);

		// Transform vector by matrix (4x4 transformation) - defined after Matrix
		static Vector3 Transform(Vector3 const& v, struct Matrix const& m);

		// Lerp and smoothstep
		static Vector3 Lerp(Vector3 const& a, Vector3 const& b, Float t)
		{
			return a + (b - a) * t;
		}

		static Vector3 SmoothStep(Vector3 const& a, Vector3 const& b, Float t)
		{
			t = t * t * (3.0f - 2.0f * t); // Smoothstep formula
			return Lerp(a, b, t);
		}

		// Common direction vectors as static const members
		static const Vector3 Zero;
		static const Vector3 One;
		static const Vector3 UnitX;
		static const Vector3 UnitY;
		static const Vector3 UnitZ;
		static const Vector3 Up;
		static const Vector3 Down;
		static const Vector3 Right;
		static const Vector3 Left;
		static const Vector3 Forward;
		static const Vector3 Backward;
	};

	inline Vector3 operator*(Float s, Vector3 const& v) { return Vector3(v.x * s, v.y * s, v.z * s); }

	struct Vector4
	{
		Float x, y, z, w;

		Vector4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
		Vector4(Float x, Float y, Float z, Float w) : x(x), y(y), z(z), w(w) {}

		Vector4 operator+(Vector4 const& v) const { return Vector4(x + v.x, y + v.y, z + v.z, w + v.w); }
		Vector4 operator-(Vector4 const& v) const { return Vector4(x - v.x, y - v.y, z - v.z, w - v.w); }
		Vector4 operator*(Float s) const { return Vector4(x * s, y * s, z * s, w * s); }
		Vector4 operator/(Float s) const { return Vector4(x / s, y / s, z / s, w / s); }

		Float Length() const { return std::sqrt(x * x + y * y + z * z + w * w); }
		Vector4 Normalized() const { Float len = Length(); return len > 0 ? *this / len : Vector4(); }
	};

	struct Matrix
	{
		Float m[4][4];

		Matrix()
		{
			for (int i = 0; i < 4; i++)
				for (int j = 0; j < 4; j++)
					m[i][j] = (i == j) ? 1.0f : 0.0f;
		}

		// Constructor from 4 Vector4s (rows)
		Matrix(Vector4 const& r0, Vector4 const& r1, Vector4 const& r2, Vector4 const& r3)
		{
			m[0][0] = r0.x; m[0][1] = r0.y; m[0][2] = r0.z; m[0][3] = r0.w;
			m[1][0] = r1.x; m[1][1] = r1.y; m[1][2] = r1.z; m[1][3] = r1.w;
			m[2][0] = r2.x; m[2][1] = r2.y; m[2][2] = r2.z; m[2][3] = r2.w;
			m[3][0] = r3.x; m[3][1] = r3.y; m[3][2] = r3.z; m[3][3] = r3.w;
		}

		Matrix operator*(Matrix const& other) const
		{
			Matrix result;
			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					result.m[i][j] = 0.0f;
					for (int k = 0; k < 4; k++)
					{
						result.m[i][j] += m[i][k] * other.m[k][j];
					}
				}
			}
			return result;
		}

		Matrix& operator*=(Matrix const& other)
		{
			*this = *this * other;
			return *this;
		}

		Matrix Transpose() const
		{
			Matrix result;
			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					result.m[i][j] = m[j][i];
				}
			}
			return result;
		}

		static const Matrix Identity;

		static Matrix CreateScale(Float sx, Float sy, Float sz)
		{
			Matrix mat;
			mat.m[0][0] = sx;
			mat.m[1][1] = sy;
			mat.m[2][2] = sz;
			return mat;
		}

		static Matrix CreateFromQuaternion(struct Quaternion const& q);

		Bool Decompose(Vector3& scale, struct Quaternion& rotation, Vector3& translation) const;
	};

	struct Quaternion
	{
		Float x, y, z, w;

		Quaternion() : x(0.0f), y(0.0f), z(0.0f), w(1.0f) {}
		Quaternion(Float x, Float y, Float z, Float w) : x(x), y(y), z(z), w(w) {}

		Quaternion operator*(Quaternion const& q) const
		{
			return Quaternion(
				w * q.x + x * q.w + y * q.z - z * q.y,
				w * q.y - x * q.z + y * q.w + z * q.x,
				w * q.z + x * q.y - y * q.x + z * q.w,
				w * q.w - x * q.x - y * q.y - z * q.z
			);
		}

		Quaternion& operator*=(Quaternion const& q)
		{
			*this = *this * q;
			return *this;
		}

		Float Length() const { return std::sqrt(x * x + y * y + z * z + w * w); }
		Float LengthSquared() const { return x * x + y * y + z * z + w * w; }

		void Normalize()
		{
			Float len = Length();
			if (len > 0.0f)
			{
				Float invLen = 1.0f / len;
				x *= invLen;
				y *= invLen;
				z *= invLen;
				w *= invLen;
			}
		}

		Quaternion Normalized() const
		{
			Float len = Length();
			if (len > 0.0f)
			{
				Float invLen = 1.0f / len;
				return Quaternion(x * invLen, y * invLen, z * invLen, w * invLen);
			}
			return Quaternion();
		}

		static Quaternion CreateFromYawPitchRoll(Float yaw, Float pitch, Float roll)
		{
			// Note: DirectXMath's XMQuaternionRotationRollPitchYaw takes (pitch, yaw, roll)
			// So when SimpleMath calls CreateFromYawPitchRoll(yaw, pitch, roll),
			// it internally calls XMQuaternionRotationRollPitchYaw(pitch, yaw, roll)
			Float halfPitch = pitch * 0.5f;
			Float halfYaw = yaw * 0.5f;
			Float halfRoll = roll * 0.5f;

			Float sinPitch = std::sin(halfPitch);
			Float cosPitch = std::cos(halfPitch);
			Float sinYaw = std::sin(halfYaw);
			Float cosYaw = std::cos(halfYaw);
			Float sinRoll = std::sin(halfRoll);
			Float cosRoll = std::cos(halfRoll);

			return Quaternion(
				cosYaw * sinPitch * cosRoll + sinYaw * cosPitch * sinRoll,
				sinYaw * cosPitch * cosRoll - cosYaw * sinPitch * sinRoll,
				cosYaw * cosPitch * sinRoll - sinYaw * sinPitch * cosRoll,
				cosYaw * cosPitch * cosRoll + sinYaw * sinPitch * sinRoll
			);
		}

		static Quaternion CreateFromAxisAngle(Vector3 const& axis, Float angle)
		{
			Float halfAngle = angle * 0.5f;
			Float sinHalf = std::sin(halfAngle);
			Float cosHalf = std::cos(halfAngle);

			return Quaternion(
				axis.x * sinHalf,
				axis.y * sinHalf,
				axis.z * sinHalf,
				cosHalf
			);
		}

		// Create a rotation from one vector to another
		static Quaternion FromToRotation(Vector3 const& from, Vector3 const& to)
		{
			Vector3 f = from.Normalized();
			Vector3 t = to.Normalized();

			Float dot = Vector3::Dot(f, t);

			if (dot >= 0.999999f) {
				return Quaternion::Identity();
			}

			if (dot <= -0.999999f) {
				// Vectors are opposite, return 180 degree rotation around any perpendicular axis
				Vector3 axis = Vector3::Cross(f, Vector3::Right);
				if (axis.LengthSquared() < 0.000001f) {
					axis = Vector3::Cross(f, Vector3::Up);
				}
				axis.Normalize();
				return CreateFromAxisAngle(axis, 3.14159265359f); // PI
			}

			Vector3 axis = Vector3::Cross(f, t);
			Float s = std::sqrt((1.0f + dot) * 2.0f);
			Float invS = 1.0f / s;

			return Quaternion(
				axis.x * invS,
				axis.y * invS,
				axis.z * invS,
				s * 0.5f
			);
		}

		// Create a rotation that looks in the forward direction with the given up vector
		static Quaternion LookRotation(Vector3 const& forward, Vector3 const& up)
		{
			Quaternion q1 = FromToRotation(Vector3::Forward, forward);

			Vector3 right = Vector3::Cross(forward, up);
			if (right.LengthSquared() < 0.000001f) {
				// forward and up are co-linear
				return q1;
			}

			Vector3 upTransformed = Vector3::Transform(Vector3::Up, q1);
			Quaternion q2 = FromToRotation(upTransformed, up);

			return q2 * q1;
		}

		static Quaternion Identity() { return Quaternion(0.0f, 0.0f, 0.0f, 1.0f); }
	};

	// Implement Vector3::Transform after Quaternion is defined
	inline Vector3 Vector3::Transform(Vector3 const& v, Quaternion const& q)
	{
		// v' = q * v * q^-1
		// For unit quaternions, q^-1 = q* (conjugate)
		// Optimized formula: v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
		Vector3 qvec(q.x, q.y, q.z);
		Vector3 cross1 = Cross(qvec, v);
		Vector3 cross2 = Cross(qvec, cross1 + v * q.w);
		return v + cross2 * 2.0f;
	}

	// Implement Vector3::Transform for Matrix
	inline Vector3 Vector3::Transform(Vector3 const& v, Matrix const& m)
	{
		Float x = v.x * m.m[0][0] + v.y * m.m[1][0] + v.z * m.m[2][0] + m.m[3][0];
		Float y = v.x * m.m[0][1] + v.y * m.m[1][1] + v.z * m.m[2][1] + m.m[3][1];
		Float z = v.x * m.m[0][2] + v.y * m.m[1][2] + v.z * m.m[2][2] + m.m[3][2];
		return Vector3(x, y, z);
	}

	// Implement Matrix functions that depend on Quaternion
	inline Matrix Matrix::CreateFromQuaternion(Quaternion const& q)
	{
		Float xx = q.x * q.x;
		Float yy = q.y * q.y;
		Float zz = q.z * q.z;
		Float xy = q.x * q.y;
		Float xz = q.x * q.z;
		Float yz = q.y * q.z;
		Float wx = q.w * q.x;
		Float wy = q.w * q.y;
		Float wz = q.w * q.z;

		Matrix mat;
		mat.m[0][0] = 1.0f - 2.0f * (yy + zz);
		mat.m[0][1] = 2.0f * (xy + wz);
		mat.m[0][2] = 2.0f * (xz - wy);
		mat.m[0][3] = 0.0f;

		mat.m[1][0] = 2.0f * (xy - wz);
		mat.m[1][1] = 1.0f - 2.0f * (xx + zz);
		mat.m[1][2] = 2.0f * (yz + wx);
		mat.m[1][3] = 0.0f;

		mat.m[2][0] = 2.0f * (xz + wy);
		mat.m[2][1] = 2.0f * (yz - wx);
		mat.m[2][2] = 1.0f - 2.0f * (xx + yy);
		mat.m[2][3] = 0.0f;

		mat.m[3][0] = 0.0f;
		mat.m[3][1] = 0.0f;
		mat.m[3][2] = 0.0f;
		mat.m[3][3] = 1.0f;

		return mat;
	}

	inline Bool Matrix::Decompose(Vector3& scale, Quaternion& rotation, Vector3& translation) const
	{
		// Extract translation
		translation.x = m[3][0];
		translation.y = m[3][1];
		translation.z = m[3][2];

		// Extract scale
		Vector3 row0(m[0][0], m[0][1], m[0][2]);
		Vector3 row1(m[1][0], m[1][1], m[1][2]);
		Vector3 row2(m[2][0], m[2][1], m[2][2]);

		scale.x = row0.Length();
		scale.y = row1.Length();
		scale.z = row2.Length();

		// Remove scaling from matrix to get rotation
		if (scale.x != 0.0f) { row0 = row0 / scale.x; }
		if (scale.y != 0.0f) { row1 = row1 / scale.y; }
		if (scale.z != 0.0f) { row2 = row2 / scale.z; }

		// Convert rotation matrix to quaternion
		Float trace = row0.x + row1.y + row2.z;
		if (trace > 0.0f)
		{
			Float s = std::sqrt(trace + 1.0f) * 2.0f;
			rotation.w = 0.25f * s;
			rotation.x = (row2.y - row1.z) / s;
			rotation.y = (row0.z - row2.x) / s;
			rotation.z = (row1.x - row0.y) / s;
		}
		else if ((row0.x > row1.y) && (row0.x > row2.z))
		{
			Float s = std::sqrt(1.0f + row0.x - row1.y - row2.z) * 2.0f;
			rotation.w = (row2.y - row1.z) / s;
			rotation.x = 0.25f * s;
			rotation.y = (row0.y + row1.x) / s;
			rotation.z = (row0.z + row2.x) / s;
		}
		else if (row1.y > row2.z)
		{
			Float s = std::sqrt(1.0f + row1.y - row0.x - row2.z) * 2.0f;
			rotation.w = (row0.z - row2.x) / s;
			rotation.x = (row0.y + row1.x) / s;
			rotation.y = 0.25f * s;
			rotation.z = (row1.z + row2.y) / s;
		}
		else
		{
			Float s = std::sqrt(1.0f + row2.z - row0.x - row1.y) * 2.0f;
			rotation.w = (row1.x - row0.y) / s;
			rotation.x = (row0.z + row2.x) / s;
			rotation.y = (row1.z + row2.y) / s;
			rotation.z = 0.25f * s;
		}

		return true;
	}

	// Define Vector3 static constants
	inline const Vector3 Vector3::Zero     = Vector3(0.0f, 0.0f, 0.0f);
	inline const Vector3 Vector3::One      = Vector3(1.0f, 1.0f, 1.0f);
	inline const Vector3 Vector3::UnitX    = Vector3(1.0f, 0.0f, 0.0f);
	inline const Vector3 Vector3::UnitY    = Vector3(0.0f, 1.0f, 0.0f);
	inline const Vector3 Vector3::UnitZ    = Vector3(0.0f, 0.0f, 1.0f);
	inline const Vector3 Vector3::Up       = Vector3(0.0f, 1.0f, 0.0f);
	inline const Vector3 Vector3::Down     = Vector3(0.0f, -1.0f, 0.0f);
	inline const Vector3 Vector3::Right    = Vector3(1.0f, 0.0f, 0.0f);
	inline const Vector3 Vector3::Left     = Vector3(-1.0f, 0.0f, 0.0f);
	inline const Vector3 Vector3::Forward  = Vector3(0.0f, 0.0f, 1.0f);
	inline const Vector3 Vector3::Backward = Vector3(0.0f, 0.0f, -1.0f);

	// Define Matrix static constants
	inline const Matrix Matrix::Identity = Matrix();

	using Color = Vector4;

	// Integer vectors
	struct Vector2u
	{
		Uint32 x, y;
		Vector2u() : x(0), y(0) {}
		Vector2u(Uint32 x, Uint32 y) : x(x), y(y) {}
	};

	struct Vector3u
	{
		Uint32 x, y, z;
		Vector3u() : x(0), y(0), z(0) {}
		Vector3u(Uint32 x, Uint32 y, Uint32 z) : x(x), y(y), z(z) {}
	};

	struct Vector4u
	{
		Uint32 x, y, z, w;
		Vector4u() : x(0), y(0), z(0), w(0) {}
		Vector4u(Uint32 x, Uint32 y, Uint32 z, Uint32 w) : x(x), y(y), z(z), w(w) {}
	};

	struct Vector2i
	{
		Int32 x, y;
		Vector2i() : x(0), y(0) {}
		Vector2i(Int32 x, Int32 y) : x(x), y(y) {}
	};

	struct Vector3i
	{
		Int32 x, y, z;
		Vector3i() : x(0), y(0), z(0) {}
		Vector3i(Int32 x, Int32 y, Int32 z) : x(x), y(y), z(z) {}
	};

	struct Vector4i
	{
		Int32 x, y, z, w;
		Vector4i() : x(0), y(0), z(0), w(0) {}
		Vector4i(Int32 x, Int32 y, Int32 z, Int32 w) : x(x), y(y), z(z), w(w) {}
	};

	struct Ray
	{
		Vector3 origin;
		Vector3 direction;

		Ray() : origin(), direction(0, 0, 1) {}
		Ray(Vector3 const& origin, Vector3 const& direction) : origin(origin), direction(direction) {}
	};
}
