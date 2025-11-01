#pragma once
#include <vector>
#include <span>
#include <algorithm>

namespace amber
{
	template<typename T>
	class CpuBuffer2D
	{
	public:
		CpuBuffer2D(Uint32 rows, Uint32 cols) : buffer(rows * cols), rows(rows), cols(cols) {}
		AMBER_DEFAULT_COPYABLE_MOVABLE(CpuBuffer2D)
		~CpuBuffer2D() = default;

		void Clear(T clear = T{})
		{
			std::fill(buffer.begin(), buffer.end(), clear);
		}
		void Resize(Uint32 r, Uint32 c)
		{
			if (rows != r || cols != c)
			{
				buffer.clear();
				buffer.resize(r * c);
				rows = r;
				cols = c;
			}
		}

		T& operator()(Uint64 row, Uint64 col) 
		{
			return buffer[row * cols + col];
		}
		T const& operator()(Uint64 row, Uint64 col) const 
		{
			return buffer[row * cols + col];
		}

#ifndef __CUDACC__
		std::span<T> operator[](Uint64 i)
		{
			return std::span{ buffer }.subspan(i * cols, cols);
		}
		std::span<T const> operator[](Uint64 i) const
		{
			return std::span{ buffer }.subspan(i * cols, cols);
		}
#endif 
		operator T* ()
		{
			return buffer.data();
		}
		operator T const* () const
		{
			return buffer.data();
		}

		T const* Data() const 
		{
			return buffer.data();
		}
		T* Data()
		{
			return buffer.data();
		}

		Uint64 Rows() const { return rows; }
		Uint64 Cols() const { return cols; }
		Uint64 Size() const { return buffer.size(); }

	private:
		std::vector<T> buffer;
		Uint64 rows;
		Uint64 cols;
	};
}