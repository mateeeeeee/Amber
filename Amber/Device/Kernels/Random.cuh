#pragma once
#include "DeviceCommon.cuh"

template<Uint32 N>
static __forceinline__ __device__  Uint32 tea(Uint32 val0, Uint32 val1)
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

// Generate random Uint32 in [0, 2^24)
static __forceinline__ __device__  Uint32 lcg(Uint32& prev)
{
	const Uint32 LCG_A = 1664525u;
	const Uint32 LCG_C = 1013904223u;
	prev = (LCG_A * prev + LCG_C);
	return prev & 0x00FFFFFF;
}

static __forceinline__ __device__  Uint32 lcg2(Uint32& prev)
{
	prev = (prev * 8121 + 28411) % 134456;
	return prev;
}

// Generate random float in [0, 1)
static __forceinline__ __device__  Float rnd(Uint32& prev)
{
	return ((Float)lcg(prev) / (Float)0x01000000);
}

static __forceinline__ __device__  Uint32 rot_seed(Uint32 seed, Uint32 frame)
{
	return seed ^ frame;
}
