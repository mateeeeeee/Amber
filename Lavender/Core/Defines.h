#pragma once
#include <cassert>

#define _LAV_STRINGIFY_IMPL(a) #a
#define _LAV_CONCAT_IMPL(x, y) x##y

#define LAV_STRINGIFY(a) _LAV_STRINGIFY_IMPL(a)
#define LAV_CONCAT(x, y) _LAV_CONCAT_IMPL( x, y )

#define LAV_ASSERT(expr)			assert(expr)
#define LAV_ASSERT_MSG(expr, msg)   assert(expr && msg)


#define LAV_NONCOPYABLE(ClassName)                 \
        ClassName(ClassName const&)            = delete; \
        ClassName& operator=(ClassName const&) = delete;

#define LAV_NONMOVABLE(ClassName)                      \
        ClassName(ClassName&&) noexcept            = delete; \
        ClassName& operator=(ClassName&&) noexcept = delete;

#define LAV_NONCOPYABLE_NONMOVABLE(ClassName) \
        LAV_NONCOPYABLE(ClassName)                \
        LAV_NONMOVABLE(ClassName)

#define LAV_DEFAULT_COPYABLE(ClassName)             \
        ClassName(ClassName const&)            = default; \
        ClassName& operator=(ClassName const&) = default;

#define LAV_DEFAULT_MOVABLE(ClassName)                  \
        ClassName(ClassName&&) noexcept            = default; \
        ClassName& operator=(ClassName&&) noexcept = default;

#define LAV_DEFAULT_COPYABLE_MOVABLE(ClassName) \
        LAV_DEFAULT_COPYABLE(ClassName)             \
        LAV_DEFAULT_MOVABLE(ClassName)
