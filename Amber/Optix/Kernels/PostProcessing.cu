#include "Color.cuh"


__global__ void PostProcess(float3* hdr_input, uchar4* ldr_output, int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;

	//int idx = y * width + x;
	//float3 color = hdr_input[idx];
	//color = aces_tonemap(color); // Apply tone mapping
	//color = pow(color, make_float3(1.0f / 2.2f)); // Gamma correction
	//ldr_output[idx] = make_uchar4(
	//	(unsigned char)(255.0f * clamp(color.x, 0.0f, 1.0f)),
	//	(unsigned char)(255.0f * clamp(color.y, 0.0f, 1.0f)),
	//	(unsigned char)(255.0f * clamp(color.z, 0.0f, 1.0f)),
	//	255);
}