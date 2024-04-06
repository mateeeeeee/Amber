cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

project(Lavender CUDA CXX)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


if(CMAKE_CONFIGURATION_TYPES)
	message("Multi-configuration generator")
	set(CMAKE_CONFIGURATION_TYPES "Debug;Release")
elseif()
	message("Single-configuration generator")
endif()

add_compile_definitions(
  "$<$<CONFIG:DEBUG>:LAVENDER_DEBUG_BUILD>"
  "$<$<CONFIG:RELEASE>: LAVENDER_RELEASE_BUILD>"
)

add_subdirectory(Libraries/pbrtParser) 
add_subdirectory(Lavender)

add_dependencies(Lavender pbrtParser)
