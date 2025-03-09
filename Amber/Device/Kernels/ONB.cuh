#pragma once 
#include "Math.cuh"
#include "DeviceTypes.cuh"


__forceinline__ __device__ void BuildONB(Float3 const& N, Float3& T, Float3& B)
{
	if (N.z < -0.99998796f)  
	{
		T = MakeFloat3(0.0f, -1.0f, 0.0f);
		B = MakeFloat3(-1.0f, 0.0f, 0.0f);
		return;
	}
	Float nxa = -N.x / (1.0f + N.z);
	T = MakeFloat3(1.0f + N.x * nxa, nxa * N.y, -N.x);
	B = MakeFloat3(T.y, 1.0f - N.y * N.y / (1.0f + N.z), -N.y);
}

__forceinline__ __device__ void BuildRotatedONB(Float3 const& N, Float3& T, Float3& B, Float basis_rotation)
{
    Float3 up = abs(N.z) < 0.9999999f ? MakeFloat3(0.0f, 0.0f, 1.0f) : MakeFloat3(1.0f, 0.0f, 0.0f);
    T = normalize(cross(up, N));
    T = T * cos(basis_rotation) + cross(N, T) * sin(basis_rotation) + N * dot(N, T) * (1.0f - cos(basis_rotation));
    B = cross(N, T);
}

__forceinline__ __device__ Float3 LocalToWorldFrame(Float3 const& N, Float3 const& V)
{
    Float3 T, B;
    BuildONB(N, T, B);
    return normalize(V.x * T + V.y * B + V.z * N);
}

__forceinline__ __device__ Float3 LocalToWorldFrame(Float3 const& T, Float3 const& B, Float3 const& N, Float3 const& V)
{
    return normalize(V.x * T + V.y * B + V.z * N);
}

/*
 * Transforms V from its space to the local space around the normal
 * The given normal is the Z axis of the local frame around the normal
 */
__forceinline__ __device__ Float3 WorldToLocalFrame(Float3 const& N, Float3 const& V)
{
    Float3 T, B;
    BuildONB(N, T, B);
    return normalize(MakeFloat3(dot(V, T), dot(V, B), dot(V, N)));
}

__forceinline__ __device__ float3 WorldToLocalFrame(Float3 const& T, Float3 const& B, Float3 const& N, Float3 const& V)
{
    return normalize(MakeFloat3(dot(V, T), dot(V, B), dot(V, N)));
}
