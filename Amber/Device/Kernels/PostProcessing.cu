#include "Color.cuh"
#include "DeviceCommon.cuh"

__device__ __forceinline__ float3 ToSRGB(float3 const& color)
{
	static constexpr float INV_GAMMA = 1.0f / 2.2f;
	float3 gamma_corrected_color = make_float3(powf(color.x, INV_GAMMA), powf(color.y, INV_GAMMA), powf(color.z, INV_GAMMA));
	return make_float3(
		color.x < 0.0031308f ? 12.92f * color.x : 1.055f * gamma_corrected_color.x - 0.055f,
		color.y < 0.0031308f ? 12.92f * color.y : 1.055f * gamma_corrected_color.y - 0.055f,
		color.z < 0.0031308f ? 12.92f * color.z : 1.055f * gamma_corrected_color.z - 0.055f);
}
__device__ __forceinline__ unsigned char QuantizeUnsigned8Bits(float x)
{
	x = clamp(x, 0.0f, 1.0f);
	static constexpr unsigned int N = (1 << 8) - 1;
	static constexpr unsigned int Np1 = (1 << 8);
	return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}
__device__ __forceinline__ uchar4 MakeUChar4(float3 const& srgb)
{
	return make_uchar4(QuantizeUnsigned8Bits(srgb.x), QuantizeUnsigned8Bits(srgb.y), QuantizeUnsigned8Bits(srgb.z), 255u);
}

__device__ float Luminance(float3 color)
{
	return dot(color, make_float3(0.2126729, 0.7151522, 0.0721750));
}

__global__ void ResolveAccumulation(Float3* hdr_output, Float3* accum_input, Int width, Int height, Int frame_index)
{
	Int x = threadIdx.x + blockIdx.x * blockDim.x;
	Int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;

	Int idx = y * width + x;
	Float3 color = accum_input[idx];
	color /= (1 + frame_index);
	hdr_output[idx] = color;
}

__global__ void Tonemap(Uchar4* ldr_output, Float3* hdr_input, Int width, Int height)
{
	Int x = threadIdx.x + blockIdx.x * blockDim.x;
	Int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;

	Int idx = y * width + x;
	Float3 color = hdr_input[idx];

	Float luma = Luminance(color);
	Float toneMappedLuma = luma / (1. + luma);
	if (luma > 1e-6)
	{
		color *= toneMappedLuma / luma;
	}
	ldr_output[idx] = MakeUChar4(ToSRGB(color));
}

namespace amber
{
	extern "C" void LaunchResolveAccumulationKernel(Float3* hdr_output, Float3* accum_input, Int width, Int height, Int frame_index)
	{
		dim3 blockDim(16, 16);
		dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
		LAUNCH_KERNEL(ResolveAccumulation, gridDim, blockDim, hdr_output, accum_input, width, height, frame_index);
	}

	extern "C" void LaunchTonemapKernel(Uchar4* ldr_output, Float3* hdr_input, Int width, Int height)
	{
		dim3 blockDim(16, 16);
		dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
		LAUNCH_KERNEL(Tonemap, gridDim, blockDim, ldr_output, hdr_input, width, height);
	}
}

