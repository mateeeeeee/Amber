#pragma once
#include <cuda_runtime.h>
#include "CudaUtil.h"

namespace lavender
{
	template<typename T>
	static LAV_DEVICE [[nodiscard]] T const& Clamp(T const& val, T const& min, T const& max)
	{
		return val < min ? min : (val > max ? max : val);
	}
	template<typename T>
	static LAV_DEVICE [[nodiscard]] T const& Min(T const& val1, T const& val2)
	{
		return val1 < val2 ? val1 : val2;
	}
	template<typename T>
	static LAV_DEVICE [[nodiscard]] T const& Max(T const& val1, T const& val2)
	{
		return val1 < val2 ? val2 : val1;
	}

	class Vec3
	{
	public:
		LAV_HOST_DEVICE Vec3() : x{}, y{}, z{} {}
		LAV_HOST_DEVICE Vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }

		LAV_HOST_DEVICE inline const Vec3& operator+() const { return *this; }
		LAV_HOST_DEVICE inline Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
		LAV_HOST_DEVICE inline float operator[](int i) const { return e[i]; }
		LAV_HOST_DEVICE inline float& operator[](int i) { return e[i]; };

		LAV_HOST_DEVICE inline Vec3& operator+=(Vec3 const& v2);
		LAV_HOST_DEVICE inline Vec3& operator-=(Vec3 const& v2);
		LAV_HOST_DEVICE inline Vec3& operator*=(Vec3 const& v2);
		LAV_HOST_DEVICE inline Vec3& operator/=(Vec3 const& v2);
		LAV_HOST_DEVICE inline Vec3& operator*=(float t);
		LAV_HOST_DEVICE inline Vec3& operator/=(float t);

		LAV_HOST_DEVICE inline float Length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
		LAV_HOST_DEVICE inline float LengthSq() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
		LAV_HOST_DEVICE inline void  Normalize();

		union
		{
			float e[3];
			struct
			{
				float x, y, z;
			};
		};
	};

	LAV_HOST_DEVICE inline void Vec3::Normalize()
	{
		float k = 1.0f / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
		e[0] *= k; e[1] *= k; e[2] *= k;
	}

	LAV_HOST_DEVICE inline Vec3 operator+(const Vec3& v1, const Vec3& v2)
	{
		return Vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
	}

	LAV_HOST_DEVICE inline Vec3 operator-(const Vec3& v1, const Vec3& v2)
	{
		return Vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
	}

	LAV_HOST_DEVICE inline Vec3 operator*(const Vec3& v1, const Vec3& v2)
	{
		return Vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
	}

	LAV_HOST_DEVICE inline Vec3 operator/(const Vec3& v1, const Vec3& v2)
	{
		return Vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
	}

	LAV_HOST_DEVICE inline Vec3 operator*(float t, const Vec3& v)
	{
		return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
	}

	LAV_HOST_DEVICE inline Vec3 operator/(Vec3 v, float t)
	{
		return Vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
	}

	LAV_HOST_DEVICE inline Vec3 operator*(const Vec3& v, float t)
	{
		return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
	}

	LAV_HOST_DEVICE inline float Dot(const Vec3& v1, const Vec3& v2)
	{
		return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
	}

	LAV_HOST_DEVICE inline Vec3 Cross(const Vec3& v1, const Vec3& v2)
	{
		return Vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
			(-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
			(v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
	}


	LAV_HOST_DEVICE inline Vec3& Vec3::operator+=(const Vec3& v)
	{
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}

	LAV_HOST_DEVICE inline Vec3& Vec3::operator*=(const Vec3& v)
	{
		e[0] *= v.e[0];
		e[1] *= v.e[1];
		e[2] *= v.e[2];
		return *this;
	}

	LAV_HOST_DEVICE inline Vec3& Vec3::operator/=(const Vec3& v)
	{
		e[0] /= v.e[0];
		e[1] /= v.e[1];
		e[2] /= v.e[2];
		return *this;
	}

	LAV_HOST_DEVICE inline Vec3& Vec3::operator-=(const Vec3& v)
	{
		e[0] -= v.e[0];
		e[1] -= v.e[1];
		e[2] -= v.e[2];
		return *this;
	}

	LAV_HOST_DEVICE inline Vec3& Vec3::operator*=(float t)
	{
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;
		return *this;
	}

	LAV_HOST_DEVICE inline Vec3& Vec3::operator/=(float t)
	{
		float k = 1.0f / t;

		e[0] *= k;
		e[1] *= k;
		e[2] *= k;
		return *this;
	}

	LAV_HOST_DEVICE inline Vec3 Normalize(Vec3 v)
	{
		return v / v.Length();
	}
}