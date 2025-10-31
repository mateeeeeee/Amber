cmake_minimum_required(VERSION 3.20)

# Set configuration types BEFORE project()
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;MinSizeRel" CACHE STRING "Available configuration types" FORCE)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

project(Amber CUDA CXX)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Don't redefine CMAKE_CONFIGURATION_TYPES here
if(CMAKE_GENERATOR MATCHES "Visual Studio")
    message("Multi-configuration generator detected")
else()
    message("Single-configuration generator detected")
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /MT")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif()

# Configuration-specific defines
add_compile_definitions(
    "$<$<CONFIG:Debug>:AMBER_DEBUG_BUILD>"
    "$<$<CONFIG:Release>:AMBER_RELEASE_BUILD>"
    "$<$<CONFIG:RelWithDebInfo>:AMBER_RELWITHDEBINFO_BUILD>"
    "$<$<CONFIG:MinSizeRel>:AMBER_MINSIZEREL_BUILD>"
)

add_subdirectory(Libraries/pbrtParser) 
add_subdirectory(Amber)

add_dependencies(Amber pbrtParser)