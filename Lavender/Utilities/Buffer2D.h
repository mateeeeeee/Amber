#pragma once
#include <vector>
#include <span>

namespace lavender
{
	template<typename T>
	class Buffer2D
	{
	public:
		Buffer2D(uint32 rows, uint32 cols) : buffer(rows * cols), rows(rows), cols(cols) {}
		LAV_DEFAULT_COPYABLE_MOVABLE(Buffer2D)
		~Buffer2D() = default;

		void Clear(T clear = T{})
		{
			std::fill(buffer.begin(), buffer.end(), clear);
		}
		void Resize(uint32 r, uint32 c)
		{
			buffer.clear();
			buffer.resize(r * c);
			rows = r;
			cols = c;
		}

		T& operator()(uint64 row, uint64 col) 
		{
			return buffer[row * cols + col];
		}
		T const& operator()(uint64 row, uint64 col) const 
		{
			return buffer[row * cols + col];
		}

#ifndef __CUDACC__
		std::span<T> operator[](uint64 i)
		{
			return std::span{ buffer }.subspan(i * cols, cols);
		}
		std::span<T const> operator[](uint64 i) const
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

		uint64 Rows() const { return rows; }
		uint64 Cols() const { return cols; }

	private:
		std::vector<T> buffer;
		uint64 rows;
		uint64 cols;
	};
}