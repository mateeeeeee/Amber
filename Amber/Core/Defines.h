#pragma once
#include <cassert>

#define _AMBER_STRINGIFY_IMPL(a) #a
#define _AMBER_CONCAT_IMPL(x, y) x##y

#define AMBER_STRINGIFY(a) _AMBER_STRINGIFY_IMPL(a)
#define AMBER_CONCAT(x, y) _AMBER_CONCAT_IMPL( x, y )

#define AMBER_ASSERT(expr)			assert(expr)
#define AMBER_ASSERT_MSG(expr, msg)   assert(expr && msg)


#define AMBER_NONCOPYABLE(ClassName)                 \
        ClassName(ClassName const&)            = delete; \
        ClassName& operator=(ClassName const&) = delete;

#define AMBER_NONMOVABLE(ClassName)                      \
        ClassName(ClassName&&) noexcept            = delete; \
        ClassName& operator=(ClassName&&) noexcept = delete;

#define AMBER_NONCOPYABLE_NONMOVABLE(ClassName) \
        AMBER_NONCOPYABLE(ClassName)                \
        AMBER_NONMOVABLE(ClassName)

#define AMBER_DEFAULT_COPYABLE(ClassName)             \
        ClassName(ClassName const&)            = default; \
        ClassName& operator=(ClassName const&) = default;

#define AMBER_DEFAULT_MOVABLE(ClassName)                  \
        ClassName(ClassName&&) noexcept            = default; \
        ClassName& operator=(ClassName&&) noexcept = default;

#define AMBER_DEFAULT_COPYABLE_MOVABLE(ClassName) \
        AMBER_DEFAULT_COPYABLE(ClassName)             \
        AMBER_DEFAULT_MOVABLE(ClassName)
