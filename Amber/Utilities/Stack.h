#pragma once
#include <vector>
#include <cassert>

namespace amber
{
	template<typename T, Uint32 Capacity>
	struct SmallStack
	{
		static constexpr Uint32 capacity = Capacity;

		T elems[capacity];
		Uint32 size = 0;

		Bool IsEmpty() const { return size == 0; }
		Bool IsFull()  const { return size >= capacity; }

		void Push(T const& t)
		{
			assert(!IsFull());
			elems[size++] = t;
		}

		T Pop()
		{
			assert(!IsEmpty());
			return elems[--size];
		}
	};

	template<typename T>
	struct GrowingStack
	{
		std::vector<T> elems;

		Bool IsEmpty() const { return elems.empty(); }
		void Push(T const& t) { elems.push_back(t); }

		T Pop()
		{
			assert(!IsEmpty());
			T top = std::move(elems.back());
			elems.pop_back();
			return top;
		}

		void Clear() { elems.clear(); }
	};
}
