#pragma once
#include "DeviceCommon.cuh"
#include "Math.cuh"

struct ColorRGBA32F
{
	__device__ ColorRGBA32F() : r(0.0f), g(0.0f), b(0.0f), a(1.0f) {}
	__device__ explicit ColorRGBA32F(Float value) : r(value), g(value), b(value), a(1.0f) {}
	__device__ ColorRGBA32F(Float r, Float g, Float b, Float a) : r(r), g(g), b(b), a(a) {}
	__device__ ColorRGBA32F(Float4 color) : r(color.x), g(color.y), b(color.z), a(color.w) {}

	__device__ void operator+=(ColorRGBA32F const& other) { r += other.r; g += other.g; b += other.b; a += other.a; }
	__device__ void operator-=(ColorRGBA32F const& other) { r -= other.r; g -= other.g; b -= other.b; a -= other.a; }
	__device__ void operator*=(ColorRGBA32F const& other) { r *= other.r; g *= other.g; b *= other.b; a *= other.a; }
	__device__ void operator*=(Float k) { r *= k; g *= k; b *= k; a *= k; }
	__device__ void operator/=(ColorRGBA32F const& other) { r /= other.r; g /= other.g; b /= other.b; a /= other.a; }
	__device__ void operator/=(Float k) { r /= k; g /= k; b /= k; a /= k; }
	__device__ Bool operator!=(ColorRGBA32F const& other) { return r != other.r || g != other.g || b != other.g || a != other.a; }

	__device__ Float Length() const { return sqrtf(this->LengthSqr()); }
	__device__ Float LengthSqr() const { return r * r + g * g + b * b + a * a; }
	__device__ Float Luminance() const { return 0.3086f * r + 0.6094f * g + 0.0820f * b; }
	__device__ void  Clamp(Float min, Float max) { r = clamp(r, min, max); g = clamp(g, min, max); b = clamp(b, min, max); a = clamp(a, min, max); }
	__device__ Bool  HasNaNs() const { return isnan(r) || isnan(g) || isnan(b) || isnan(a); }
	__device__ Bool  IsBlack() const { return !(r > 0.0f || g > 0.0f || b > 0.0f); }
	__device__ Bool  IsWhite() const { return r == 1.0f && g == 1.0f && b == 1.0f; }

	__device__ Float MaxComponent() const { return max(r, max(g, b)); }
	__device__ ColorRGBA32F Normalized() const { Float length = sqrtf(r * r + g * g + b * b); return ColorRGBA32F(r / length, g / length, b / length, a /length); }

	__device__ static ColorRGBA32F Max(ColorRGBA32F const& a, ColorRGBA32F const& b) { return ColorRGBA32F(max(a.r, b.r), max(a.g, b.g), max(a.b, b.b), max(a.a, b.a)); }
	__device__ static ColorRGBA32F Min(ColorRGBA32F const& a, ColorRGBA32F const& b) { return ColorRGBA32F(min(a.r, b.r), min(a.g, b.g), min(a.b, b.b), min(a.a, b.a)); }

	__device__ Float& operator[](Uint index) { return *(&r + index); }
	__device__ Float operator[](Uint index) const { return *(&r + index); }
	__device__ explicit operator Float4() const
	{
		return MakeFloat4(r, g, b, a);
	}

	Float r, g, b, a;
};

__device__ __forceinline__ ColorRGBA32F operator+ (ColorRGBA32F const& a, ColorRGBA32F const& b) { return ColorRGBA32F(a.r + b.r, a.g + b.g, a.b + b.b, a.a + b.a); }
__device__ __forceinline__ ColorRGBA32F operator- (ColorRGBA32F const& c) { return ColorRGBA32F(-c.r, -c.g, -c.b, c.a); }
__device__ __forceinline__ ColorRGBA32F operator- (ColorRGBA32F const& a, ColorRGBA32F const& b) { return ColorRGBA32F(a.r - b.r, a.g - b.g, a.b - b.b, a.a - b.a); }
__device__ __forceinline__ ColorRGBA32F operator* (ColorRGBA32F const& a, ColorRGBA32F const& b) { return ColorRGBA32F(a.r * b.r, a.g * b.g, a.b * b.b, a.a * b.a); }
__device__ __forceinline__ ColorRGBA32F operator* (Float k, ColorRGBA32F const& c) { return ColorRGBA32F(c.r * k, c.g * k, c.b * k, c.a * k); }
__device__ __forceinline__ ColorRGBA32F operator* (ColorRGBA32F const& c, Float k) { return ColorRGBA32F(c.r * k, c.g * k, c.b * k, c.a * k); }
__device__ __forceinline__ ColorRGBA32F operator/ (ColorRGBA32F const& a, ColorRGBA32F const& b) { return ColorRGBA32F(a.r / b.r, a.g / b.g, a.b / b.b, a.a / b.a); }
__device__ __forceinline__ ColorRGBA32F operator/ (Float k, ColorRGBA32F const& c) { return ColorRGBA32F(k / c.r, k / c.g, k / c.b, k / c.a); }
__device__ __forceinline__ ColorRGBA32F operator/ (ColorRGBA32F const& c, Float k) { return ColorRGBA32F(c.r / k, c.g / k, c.b / k, c.a / k); }
__device__ __forceinline__ ColorRGBA32F sqrt(ColorRGBA32F const& col) { return ColorRGBA32F(sqrtf(col.r), sqrtf(col.g), sqrtf(col.b), sqrtf(col.a)); }
__device__ __forceinline__ ColorRGBA32F exp(ColorRGBA32F const& col) { return ColorRGBA32F(expf(col.r), expf(col.g), expf(col.b), expf(col.a)); }
__device__ __forceinline__ ColorRGBA32F log(ColorRGBA32F const& col) { return ColorRGBA32F(logf(col.r), logf(col.g), logf(col.b), logf(col.a)); }
__device__ __forceinline__ ColorRGBA32F pow(ColorRGBA32F const& col, Float k) { return ColorRGBA32F(powf(col.r, k), powf(col.g, k), powf(col.b, k), powf(col.a, k)); }


struct ColorRGB32F
{
	__device__ constexpr ColorRGB32F() : r(0.0f), g(0.0f), b(0.0f) {}
	__device__ constexpr explicit ColorRGB32F(Float value) : r(value), g(value), b(value) {}
	__device__ constexpr ColorRGB32F(Float r, Float g, Float b) : r(r), g(g), b(b) {}
	__device__ constexpr ColorRGB32F(Float3 color) : r(color.x), g(color.y), b(color.z) {}

	__device__ void operator+=(ColorRGB32F const& other) { r += other.r; g += other.g; b += other.b; }
	__device__ void operator-=(ColorRGB32F const& other) { r -= other.r; g -= other.g; b -= other.b; }
	__device__ void operator*=(ColorRGB32F const& other) { r *= other.r; g *= other.g; b *= other.b; }
	__device__ void operator*=(Float k) { r *= k; g *= k; b *= k; }
	__device__ void operator/=(ColorRGB32F const& other) { r /= other.r; g /= other.g; b /= other.b; }
	__device__ void operator/=(Float k) { r /= k; g /= k; b /= k; }
	__device__ Bool operator!=(ColorRGB32F const& other) { return r != other.r || g != other.g || b != other.g; }

	__device__ Float Length() const { return sqrtf(this->LengthSqr()); }
	__device__ Float LengthSqr() const { return r * r + g * g + b * b; }
	__device__ Float Luminance() const { return 0.3086f * r + 0.6094f * g + 0.0820f * b; }
	__device__ void  Clamp(Float min, Float max) { r = clamp(r, min, max); g = clamp(g, min, max); b = clamp(b, min, max); }
	__device__ Bool  HasNaNs() const { return isnan(r) || isnan(g) || isnan(b); }
	__device__ Bool  IsBlack() const { return !(r > 0.0f || g > 0.0f || b > 0.0f); }
	__device__ Bool  IsWhite() const { return r == 1.0f && g == 1.0f && b == 1.0f; }

	__device__ Float MaxComponent() const { return max(r, max(g, b)); }
	__device__ ColorRGB32F Normalized() const { float length = sqrtf(r * r + g * g + b * b); return ColorRGB32F(r / length, g / length, b / length); }

	__device__ static ColorRGB32F Max(ColorRGB32F const& a, ColorRGB32F const& b) { return ColorRGB32F(max(a.r, b.r), max(a.g, b.g), max(a.b, b.b)); }
	__device__ static ColorRGB32F Min(ColorRGB32F const& a, ColorRGB32F const& b) { return ColorRGB32F(min(a.r, b.r), min(a.g, b.g), min(a.b, b.b)); }
	__device__ Float& operator[](Uint index) { return *(&r + index); }
	__device__ Float operator[](Uint index) const { return *(&r + index); }
	__device__ explicit operator Float3() const
	{
		return MakeFloat3(r, g, b);
	}

	Float r, g, b;
};
static constexpr ColorRGB32F ColorRGB32F_Black(0.0f);
static constexpr ColorRGB32F ColorRGB32F_White(1.0f);

__device__ __forceinline__ ColorRGB32F operator+ (ColorRGB32F const& a, ColorRGB32F const& b) { return ColorRGB32F(a.r + b.r, a.g + b.g, a.b + b.b); }
__device__ __forceinline__ ColorRGB32F operator- (ColorRGB32F const& c) { return ColorRGB32F(-c.r, -c.g, -c.b); }
__device__ __forceinline__ ColorRGB32F operator- (ColorRGB32F const& a, ColorRGB32F const& b) { return ColorRGB32F(a.r - b.r, a.g - b.g, a.b - b.b); }
__device__ __forceinline__ ColorRGB32F operator* (ColorRGB32F const& a, ColorRGB32F const& b) { return ColorRGB32F(a.r * b.r, a.g * b.g, a.b * b.b); }
__device__ __forceinline__ ColorRGB32F operator* (Float k, ColorRGB32F const& c) { return ColorRGB32F(c.r * k, c.g * k, c.b * k); }
__device__ __forceinline__ ColorRGB32F operator* (ColorRGB32F const& c, Float k) { return ColorRGB32F(c.r * k, c.g * k, c.b * k); }
__device__ __forceinline__ ColorRGB32F operator/ (ColorRGB32F const& a, ColorRGB32F const& b) { return ColorRGB32F(a.r / b.r, a.g / b.g, a.b / b.b); }
__device__ __forceinline__ ColorRGB32F operator/ (Float k, ColorRGB32F const& c) { return ColorRGB32F(k / c.r, k / c.g, k / c.b); }
__device__ __forceinline__ ColorRGB32F operator/ (ColorRGB32F const& c, Float k) { return ColorRGB32F(c.r / k, c.g / k, c.b / k); }
__device__ __forceinline__ ColorRGB32F sqrt(ColorRGB32F const& col) { return ColorRGB32F(sqrtf(col.r), sqrtf(col.g), sqrtf(col.b)); }
__device__ __forceinline__ ColorRGB32F exp(ColorRGB32F const& col) { return ColorRGB32F(expf(col.r), expf(col.g), expf(col.b)); }
__device__ __forceinline__ ColorRGB32F log(ColorRGB32F const& col) { return ColorRGB32F(logf(col.r), logf(col.g), logf(col.b)); }
__device__ __forceinline__ ColorRGB32F pow(ColorRGB32F const& col, Float k) { return ColorRGB32F(powf(col.r, k), powf(col.g, k), powf(col.b, k)); }
__device__ __forceinline__ ColorRGB32F lerp(ColorRGB32F const& a, ColorRGB32F const& b, Float t) { return a * (1.0f - t) + b * t; }

struct ColorRGBA8
{
	__device__ ColorRGBA8() : r(0), g(0), b(0), a(255) {}
	__device__ explicit ColorRGBA8(Uchar v) : r(v), g(v), b(v), a(255) {}

	__device__ ColorRGBA8(Uchar r, Uchar g, Uchar b, Uchar a) : r(r), g(g), b(b), a(a) {}
	__device__ ColorRGBA8(Uchar4 color) : r(color.x), g(color.y), b(color.z), a(color.w) {}
	__device__ ColorRGBA8(Float3 color) : r(QuantizeUnsigned8Bits(color.x)), g(QuantizeUnsigned8Bits(color.y)), b(QuantizeUnsigned8Bits(color.z)), a(255) {}

	__device__ void operator+=(ColorRGBA8 const& other) { r += other.r; g += other.g; b += other.b; a += other.a; }
	__device__ void operator-=(ColorRGBA8 const& other) { r -= other.r; g -= other.g; b -= other.b; a -= other.a; }
	__device__ void operator*=(ColorRGBA8 const& other) { r *= other.r; g *= other.g; b *= other.b; a *= other.a; }
	__device__ void operator*=(Float k) { r *= k; g *= k; b *= k; a *= k; }
	__device__ void operator/=(ColorRGBA8 const& other) { r /= other.r; g /= other.g; b /= other.b; a /= other.a; }
	__device__ void operator/=(Float k) { r /= k; g /= k; b /= k; a /= k; }
	__device__ Bool operator!=(ColorRGBA8 const& other) { return r != other.r || g != other.g || b != other.g || a != other.a; }

	__device__ void  Clamp(Uchar min, Uchar max) { r = clamp(r, min, max); g = clamp(g, min, max); b = clamp(b, min, max); a = clamp(a, min, max); }
	__device__ Bool  IsBlack() const { return !(r > 0 || g > 0 || b > 0); }
	__device__ Bool  IsWhite() const { return r == 255 && g == 255 && b == 255; }

	__device__ Float MaxComponent() const { return max(r, max(g, b)); }
	__device__ static ColorRGBA8 Max(ColorRGBA8 const& a, ColorRGBA8 const& b) { return ColorRGBA8(max(a.r, b.r), max(a.g, b.g), max(a.b, b.b), 255); }
	__device__ static ColorRGBA8 Min(ColorRGBA8 const& a, ColorRGBA8 const& b) { return ColorRGBA8(min(a.r, b.r), min(a.g, b.g), min(a.b, b.b), 255); }
	__device__ static Uchar QuantizeUnsigned8Bits(Float x)
	{
		x = clamp(x, 0.0f, 1.0f);
		static constexpr Uint N = (1 << 8) - 1;
		static constexpr Uint Np1 = (1 << 8);
		return (Uchar)min((Uint)(x * (Float)Np1), (Uint)N);
	}

	__device__ Uchar& operator[](Uint index) { return *(&r + index); }
	__device__ Uchar operator[](Uint index) const { return *(&r + index); }
	__device__ explicit operator Uchar4() const
	{
		return MakeUchar4(r, g, b, a);
	}

	Uchar r, g, b, a;
};
__device__ __forceinline__ ColorRGBA8 SRGB(ColorRGB32F const& color)
{
	static constexpr Float INV_GAMMA = 1.0f / 2.2f;
	return MakeFloat3(
		color.r < 0.0031308f ? 12.92f * color.r : 1.055f * powf(color.r, INV_GAMMA) - 0.055f,
		color.g < 0.0031308f ? 12.92f * color.g : 1.055f * powf(color.g, INV_GAMMA) - 0.055f,
		color.b < 0.0031308f ? 12.92f * color.b : 1.055f * powf(color.b, INV_GAMMA) - 0.055f);
}
__device__ __forceinline__ ColorRGBA8 SRGB(ColorRGBA32F const& color)
{
	return SRGB(ColorRGB32F(color.r, color.g, color.b));
}

__device__ __forceinline__ ColorRGBA8 operator+ (ColorRGBA8 const& a, ColorRGBA8 const& b) { return ColorRGBA8(a.r + b.r, a.g + b.g, a.b + b.b, a.a + b.a); }
__device__ __forceinline__ ColorRGBA8 operator- (ColorRGBA8 const& c) { return ColorRGBA8(-c.r, -c.g, -c.b, c.a); }
__device__ __forceinline__ ColorRGBA8 operator- (ColorRGBA8 const& a, ColorRGBA8 const& b) { return ColorRGBA8(a.r - b.r, a.g - b.g, a.b - b.b, a.a - b.a); }
__device__ __forceinline__ ColorRGBA8 operator* (ColorRGBA8 const& a, ColorRGBA8 const& b) { return ColorRGBA8(a.r * b.r, a.g * b.g, a.b * b.b, a.a * b.a); }
__device__ __forceinline__ ColorRGBA8 operator* (Float k, ColorRGBA8 const& c) { return ColorRGBA8(c.r * k, c.g * k, c.b * k, c.a * k); }
__device__ __forceinline__ ColorRGBA8 operator* (ColorRGBA8 const& c, Float k) { return ColorRGBA8(c.r * k, c.g * k, c.b * k, c.a * k); }
__device__ __forceinline__ ColorRGBA8 operator/ (ColorRGBA8 const& a, ColorRGBA8 const& b) { return ColorRGBA8(a.r / b.r, a.g / b.g, a.b / b.b, a.a / b.a); }
__device__ __forceinline__ ColorRGBA8 operator/ (Float k, ColorRGBA8 const& c) { return ColorRGBA8(k / c.r, k / c.g, k / c.b, k / c.a); }
__device__ __forceinline__ ColorRGBA8 operator/ (ColorRGBA8 const& c, Float k) { return ColorRGBA8(c.r / k, c.g / k, c.b / k, c.a / k); }
__device__ __forceinline__ ColorRGBA8 sqrt(ColorRGBA8 const& col) { return ColorRGBA8(sqrtf(col.r), sqrtf(col.g), sqrtf(col.b), sqrtf(col.a)); }
__device__ __forceinline__ ColorRGBA8 exp(ColorRGBA8 const& col) { return ColorRGBA8(expf(col.r), expf(col.g), expf(col.b), expf(col.a)); }
__device__ __forceinline__ ColorRGBA8 log(ColorRGBA8 const& col) { return ColorRGBA8(logf(col.r), logf(col.g), logf(col.b), logf(col.a)); }
__device__ __forceinline__ ColorRGBA8 pow(ColorRGBA8 const& col, Float k) { return ColorRGBA8(powf(col.r, k), powf(col.g, k), powf(col.b, k), powf(col.a, k)); }
