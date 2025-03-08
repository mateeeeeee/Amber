#include "Color.cuh"
#include "DeviceCommon.cuh"

__global__ void ResolveAccumulation(float3* hdr_output, float3* accum_input, int width, int height, int frame_index)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;

	int idx = y * width + x;
	float3 color = accum_input[idx];
	color /= (1 + frame_index);
	hdr_output[idx] = color;
}

__global__ void Tonemap(uchar4* ldr_output, float3* hdr_input, int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;

	int idx = y * width + x;
	float3 color = hdr_input[idx];

	float luma = Luminance(color);
	float toneMappedLuma = luma / (1. + luma);
	if (luma > 1e-6)
	{
		color *= toneMappedLuma / luma;
	}

	ldr_output[idx] = MakeUChar4(ToSRGB(color));
}


extern "C" void LaunchResolveAccumulationKernel(float3* hdr_output, float3* accum_input, int width, int height, int frame_index)
{
	dim3 blockDim(16, 16);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
	LAUNCH_KERNEL(ResolveAccumulation, gridDim, blockDim, hdr_output, accum_input, width, height, frame_index);
}

extern "C" void LaunchTonemapKernel(uchar4* ldr_output, float3* hdr_input, int width, int height)
{
	dim3 blockDim(16, 16);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
	LAUNCH_KERNEL(Tonemap, gridDim, blockDim, ldr_output, hdr_input, width, height);
}
