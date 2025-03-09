#include "Color.cuh"
#include "DeviceCommon.cuh"

__global__ void ResolveAccumulation(Float3* hdr_output, Float3* accum_input, Uint width, Uint height, Uint frame_index)
{
	Uint x = threadIdx.x + blockIdx.x * blockDim.x;
	Uint y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;

	Uint idx = y * width + x;
	ColorRGB32F color(accum_input[idx]);
	color /= (1 + frame_index);
	hdr_output[idx] = static_cast<Float3>(color);
}

__global__ void Tonemap(Uchar4* ldr_output, Float3* hdr_input, Uint width, Uint height)
{
	Uint x = threadIdx.x + blockIdx.x * blockDim.x;
	Uint y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;

	Uint idx = y * width + x;
	ColorRGB32F color(hdr_input[idx]);

	Float luma = color.Luminance();
	Float tone_mapped_luma = luma / (1. + luma);
	if (luma > 1e-6)
	{
		color *= tone_mapped_luma / luma;
	}
	ldr_output[idx] = static_cast<Uchar4>(SRGB(color));
}

namespace amber
{
	extern "C" void LaunchResolveAccumulationKernel(Float3* hdr_output, Float3* accum_input, Uint width, Uint height, Uint frame_index)
	{
		dim3 block_dim(16, 16);
		dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y);
		LAUNCH_KERNEL(ResolveAccumulation, grid_dim, block_dim, hdr_output, accum_input, width, height, frame_index);
	}

	extern "C" void LaunchTonemapKernel(Uchar4* ldr_output, Float3* hdr_input, Uint width, Uint height)
	{
		dim3 block_dim(16, 16);
		dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y);
		LAUNCH_KERNEL(Tonemap, grid_dim, block_dim, ldr_output, hdr_input, width, height);
	}
}

