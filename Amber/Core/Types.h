#pragma once
#include <cstdint>

#ifndef __CUDACC__
#define STD std::
#else
#define STD 
#endif

namespace amber
{
	using Uint8		= STD uint8_t;
	using Uint16	= STD uint16_t;
	using Uint32	= STD uint32_t;
	using Uint64	= STD uint64_t;
	using Int8		= STD int8_t;
	using Int16		= STD int16_t;
	using Int32		= STD int32_t;
	using Int64		= STD int64_t;
	using Bool32	= STD uint32_t;
	using Bool		= bool;
	using Char		= char;
	using Uchar		= unsigned char;
	using Short		= short;
	using Ushort	= unsigned short;
	using Int		= int;
	using Uint		= unsigned int;
	using Float		= float;
	using Float64	= double;
	using Uintptr	= STD uintptr_t;
	using Intptr	= STD intptr_t;
}