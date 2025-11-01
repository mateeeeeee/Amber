#pragma once
#include "DeviceCommon.cuh"

template<Uint32 N>
static __forceinline__ __device__ Uint32 Tea(Uint32 val0, Uint32 val1)
{
	Uint32 v0 = val0;
	Uint32 v1 = val1;
	Uint32 s0 = 0;
	for (Uint32 n = 0; n < N; n++)
	{
		s0 += 0x9e3779b9;
		v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
		v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
	}

	return v0;
}
static __forceinline__ __device__ Uint32 PCG(Uint32 input)
{
	Uint32 state = input * 747796405u + 2891336453u;
	Uint32 word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

// Generate random Uint32 in [0, 2^24)
static __forceinline__ __device__  Uint32 LCG(Uint32& prev)
{
	constexpr Uint32 LCG_A = 1664525u;
	constexpr Uint32 LCG_C = 1013904223u;
	prev = (LCG_A * prev + LCG_C);
	return prev & 0x00FFFFFF;
}

struct PRNG
{
	__device__ static PRNG Create(Uint32 val0, Uint32 val1)
	{
		PRNG rng;
		rng.seed = Tea<4>(val0, val1);
		return rng;
	}

	__device__ Uint32 RandomInt()
	{
		seed = PCG(seed);
		return seed;
	}

	__device__ Float RandomFloat()
	{
		return ((Float)LCG(seed) / (Float)0x01000000);
	}

	__device__ Float2 RandomFloat2()
	{
		return MakeFloat2(RandomFloat(), RandomFloat());
	}
	__device__ Float3 RandomFloat3()
	{
		return MakeFloat3(RandomFloat(), RandomFloat(), RandomFloat());
	}

	Uint32 seed;
};