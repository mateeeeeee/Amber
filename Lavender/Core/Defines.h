#pragma once
#include <cassert>

#define _LAVENDER_STRINGIFY_IMPL(a) #a
#define _LAVENDER_CONCAT_IMPL(x, y) x##y

#define LAVENDER_STRINGIFY(a) _LAVENDER_STRINGIFY_IMPL(a)
#define LAVENDER_CONCAT(x, y) _LAVENDER_CONCAT_IMPL( x, y )

#define LAVENDER_ASSERT(expr)			assert(expr)
#define LAVENDER_ASSERT_MSG(expr, msg)  assert(expr && msg)
#define LAVENDER_OPTIMIZE_ON			pragma optimize("", on)
#define LAVENDER_OPTIMIZE_OFF			pragma optimize("", off)
#define LAVENDER_WARNINGS_OFF			pragma(warning(push, 0))
#define LAVENDER_WARNINGS_ON			pragma(warning(pop))
#define LAVENDER_DEBUGBREAK()			__debugbreak()
#define LAVENDER_FORCEINLINE			__forceinline
#define LAVENDER_INLINE				    inline
#define LAVENDER_NODISCARD				[[nodiscard]]
#define LAVENDER_NORETURN				[[noreturn]]
#define LAVENDER_DEPRECATED			    [[deprecated]]
#define LAVENDER_DEPRECATED_MSG(msg)	[[deprecated(#msg)]]
#define LAVENDER_ALIGN(align)           alignas(align) 

#ifdef __GNUC__ 
#define LAVENDER_UNREACHABLE()			___builtin_unreachable();
#elifdef _MSC_VER
#define LAVENDER_UNREACHABLE()			___assume(false);
#else
#define LAVENDER_UNREACHABLE()	
#endif



#define LAVENDER_NONCOPYABLE(ClassName)                 \
    ClassName(ClassName const&)            = delete; \
    ClassName& operator=(ClassName const&) = delete;

#define LAVENDER_NONMOVABLE(ClassName)                      \
    ClassName(ClassName&&) noexcept            = delete; \
    ClassName& operator=(ClassName&&) noexcept = delete;

#define LAVENDER_NONCOPYABLE_NONMOVABLE(ClassName) \
        LAVENDER_NONCOPYABLE(ClassName)                \
        LAVENDER_NONMOVABLE(ClassName)

#define LAVENDER_DEFAULT_COPYABLE(ClassName)             \
    ClassName(ClassName const&)            = default; \
    ClassName& operator=(ClassName const&) = default;

#define LAVENDER_DEFAULT_MOVABLE(ClassName)                  \
    ClassName(ClassName&&) noexcept            = default; \
    ClassName& operator=(ClassName&&) noexcept = default;

#define LAVENDER_DEFAULT_COPYABLE_MOVABLE(ClassName) \
    LAVENDER_DEFAULT_COPYABLE(ClassName)             \
    LAVENDER_DEFAULT_MOVABLE(ClassName)
