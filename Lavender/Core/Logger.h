#pragma once
#include "spdlog/spdlog.h"

namespace wave
{
#define LAVENDER_DEBUG(fmt, ...)  spdlog::debug(fmt, __VA_ARGS__)
#define LAVENDER_INFO(fmt, ...)   spdlog::info(fmt, __VA_ARGS__)
#define LAVENDER_WARN(fmt, ...)   spdlog::warn(fmt, __VA_ARGS__)
#define LAVENDER_ERROR(fmt, ...)  spdlog::error(fmt, __VA_ARGS__)
}