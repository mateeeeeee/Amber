#pragma once
#include <cassert>

#define _AMBER_STRINGIFY_IMPL(a) #a
#define _AMBER_CONCAT_IMPL(x, y) x##y

#define AMBER_STRINGIFY(a) _AMBER_STRINGIFY_IMPL(a)
#define AMBER_CONCAT(x, y) _AMBER_CONCAT_IMPL( x, y )

#define AMBER_ASSERT(expr)			assert(expr)
#define AMBER_ASSERT_MSG(expr, msg)   assert(expr && msg)
#define AMBER_UNREACHABLE()			__assume(false)
#define AMBER_FORCEINLINE			    __forceinline
#define AMBER_NODISCARD				[[nodiscard]]
#define AMBER_NORETURN				[[noreturn]]
#define AMBER_DEPRECATED			    [[deprecated]]
#define AMBER_MAYBE_UNUSED            [[maybe_unused]]
#define AMBER_DEPRECATED_MSG(msg)	    [[deprecated(#msg)]]
#define AMBER_DEBUGZONE_BEGIN         __pragma(optimize("", off))
#define AMBER_DEBUGZONE_END           __pragma(optimize("", on))

#if defined(_MSC_VER)
#define AMBER_UNREACHABLE()			__assume(false)
#define AMBER_DEBUGBREAK()			__debugbreak()
#elif defined(__GNUC__)
#define AMBER_UNREACHABLE()			__builtin_unreachable()
#define AMBER_DEBUGBREAK()	        __builtin_trap()
#else 
#define AMBER_UNREACHABLE()			
#define AMBER_DEBUGBREAK()	        
#endif

#define AMBER_NONCOPYABLE(Class)                 \
        Class(Class const&)            = delete; \
        Class& operator=(Class const&) = delete;

#define AMBER_NONMOVABLE(Class)                      \
        Class(Class&&) noexcept            = delete; \
        Class& operator=(Class&&) noexcept = delete;

#define AMBER_NONCOPYABLE_NONMOVABLE(Class) \
        AMBER_NONCOPYABLE(Class)                \
        AMBER_NONMOVABLE(Class)

#define AMBER_DEFAULT_COPYABLE(Class)             \
        Class(Class const&)            = default; \
        Class& operator=(Class const&) = default;

#define AMBER_DEFAULT_MOVABLE(Class)                  \
        Class(Class&&) noexcept            = default; \
        Class& operator=(Class&&) noexcept = default;

#define AMBER_DEFAULT_COPYABLE_MOVABLE(Class) \
        AMBER_DEFAULT_COPYABLE(Class)             \
        AMBER_DEFAULT_MOVABLE(Class)
