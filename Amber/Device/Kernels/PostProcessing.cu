#include "Color.cuh"
#include "DeviceCommon.cuh"

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

