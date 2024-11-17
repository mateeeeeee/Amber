#include "Color.cuh"


__global__ void GammaCorrection(float3* ldr_output, float3* hdr_input, int width, int height, int frame_index)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;

	int idx = y * width + x;
	float3 color = hdr_input[idx];
	color /= (1 + frame_index);
	ldr_output[idx] = ToSRGB(color);
}

__global__ void ConvertLDRBuffer(uchar4* ldr_output, float3* ldr_input, int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;

	int idx = y * width + x;
	ldr_output[idx] = MakeUChar4(ldr_input[idx]);
}

extern "C" void LaunchGammaCorrectionKernel(float3* ldr_output, float3* hdr_input, int width, int height, int frame_index)
{
	dim3 blockDim(16, 16);  
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);  
	GammaCorrection<<<gridDim, blockDim>>>(ldr_output, hdr_input, width, height, frame_index);
}

extern "C" void LaunchConvertLDRBufferKernel(uchar4* ldr_output, float3* hdr_input, int width, int height)
{
	dim3 blockDim(16, 16);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
	ConvertLDRBuffer<<<gridDim, blockDim>>>(ldr_output, hdr_input, width, height);
}