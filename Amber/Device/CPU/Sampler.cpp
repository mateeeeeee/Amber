#include "Sampler.h"
#include <cmath>
#include <algorithm>

namespace amber
{
	Vector4 Sampler::Sample(Image const& image, Vector2 uv) const
	{
		Int32 w = image.GetWidth();
		Int32 h = image.GetHeight();
		Uint8 const* data = image.GetData();

		// Only repeat wrap supported for now
		uv.x = uv.x - std::floor(uv.x);
		uv.y = uv.y - std::floor(uv.y);

		auto FetchTexel = [&](Int32 x, Int32 y) -> Vector4
		{
			x = std::clamp(x, 0, w - 1);
			y = std::clamp(y, 0, h - 1);
			Uint8 const* p = data + (y * w + x) * 4;
			Vector4 color(p[0] / 255.0f, p[1] / 255.0f, p[2] / 255.0f, p[3] / 255.0f);
			if (image.IsSRGB())
			{
				color.x = std::pow(color.x, 2.2f);
				color.y = std::pow(color.y, 2.2f);
				color.z = std::pow(color.z, 2.2f);
			}
			return color;
		};

		if (filter == FilterMode::Nearest)
		{
			Int32 x = static_cast<Int32>(uv.x * w);
			Int32 y = static_cast<Int32>(uv.y * h);
			return FetchTexel(x, y);
		}

		// Bilinear (default, also used as fallback for trilinear/anisotropic)
		Float fx = uv.x * w - 0.5f;
		Float fy = uv.y * h - 0.5f;
		Int32 x0 = static_cast<Int32>(std::floor(fx));
		Int32 y0 = static_cast<Int32>(std::floor(fy));
		Int32 x1 = x0 + 1;
		Int32 y1 = y0 + 1;

		// Repeat wrap for neighbouring texels
		x0 = ((x0 % w) + w) % w;
		x1 = ((x1 % w) + w) % w;
		y0 = ((y0 % h) + h) % h;
		y1 = ((y1 % h) + h) % h;

		Float tx = fx - std::floor(fx);
		Float ty = fy - std::floor(fy);

		Vector4 c00 = FetchTexel(x0, y0);
		Vector4 c10 = FetchTexel(x1, y0);
		Vector4 c01 = FetchTexel(x0, y1);
		Vector4 c11 = FetchTexel(x1, y1);

		Vector4 c0 = c00 * (1.0f - tx) + c10 * tx;
		Vector4 c1 = c01 * (1.0f - tx) + c11 * tx;
		return c0 * (1.0f - ty) + c1 * ty;
	}
}
