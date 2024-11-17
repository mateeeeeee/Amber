#include "Color.cuh"


__global__ void PostProcess(uchar4* ldr_output, float4* hdr_input, int width, int height, int frame_index)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;

	int idx = y * width + x;
	float4 color = hdr_input[idx];
	color /= (1 + frame_index);

	ldr_output[idx] = MakeColor(make_float3(color.x, color.y, color.z));
}

extern "C" void LaunchPostProcessKernel(uchar4* ldr_output, float4* hdr_input, int width, int height, int frame_index)
{
	dim3 blockDim(16, 16);  
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);  
	PostProcess<<<gridDim, blockDim>>>(ldr_output, hdr_input, width, height, frame_index);
}