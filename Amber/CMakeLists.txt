if(NOT APPLE)
    find_package(CUDA REQUIRED)
    find_package(CUDAToolkit REQUIRED)

    # Can be set via: cmake -DOptiX_INSTALL_DIR=/path/to/optix ..
    if(NOT DEFINED OptiX_INSTALL_DIR)
        if(WIN32)
            file(GLOB OPTIX_PATHS "C:/ProgramData/NVIDIA Corporation/OptiX SDK*")
        else()
            file(GLOB OPTIX_PATHS
                "/opt/optix*"
                "/usr/local/optix*"
                "$ENV{HOME}/optix*"
                "$ENV{HOME}/Downloads/NVIDIA-OptiX-SDK*")
        endif()

        if(OPTIX_PATHS)
            list(SORT OPTIX_PATHS)
            list(REVERSE OPTIX_PATHS)
            list(GET OPTIX_PATHS 0 OptiX_INSTALL_DIR)
            message(STATUS "Found OptiX SDK at: ${OptiX_INSTALL_DIR}")
        else()
            message(FATAL_ERROR "OptiX SDK not found. Please set OptiX_INSTALL_DIR via cmake -DOptiX_INSTALL_DIR=/path/to/optix")
        endif()
    endif()

    if(NOT EXISTS "${OptiX_INSTALL_DIR}/include/optix.h")
        message(FATAL_ERROR "OptiX not found at ${OptiX_INSTALL_DIR}. Please set OptiX_INSTALL_DIR to the correct path.")
    endif()

    file(STRINGS "${OptiX_INSTALL_DIR}/include/optix.h" OPTIX_VERSION_LINE REGEX "^#define OPTIX_VERSION")
    if(OPTIX_VERSION_LINE)
        string(REGEX MATCH "[0-9]+" OPTIX_VERSION "${OPTIX_VERSION_LINE}")
        math(EXPR OPTIX_VERSION_MAJOR "${OPTIX_VERSION} / 10000")
        math(EXPR OPTIX_VERSION_MINOR "(${OPTIX_VERSION} % 10000) / 100")
        math(EXPR OPTIX_VERSION_PATCH "${OPTIX_VERSION} % 100")
        message(STATUS "OptiX version: ${OPTIX_VERSION_MAJOR}.${OPTIX_VERSION_MINOR}.${OPTIX_VERSION_PATCH}")

        if(OPTIX_VERSION_MAJOR LESS 8)
            message(FATAL_ERROR "OptiX 8.0 or newer required. Found version ${OPTIX_VERSION_MAJOR}.${OPTIX_VERSION_MINOR}.${OPTIX_VERSION_PATCH}")
        endif()
    else()
        message(WARNING "Could not determine OptiX version")
    endif()

    set(OPTIX_INSTALL_DIR "${OptiX_INSTALL_DIR}")

    find_library(NVRTC_LIB nvrtc PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

    if(NOT NVRTC_LIB)
        message(FATAL_ERROR "NVRTC library not found!")
    endif()
endif()


set(CORE_FILES
    Core/Types.h
    Core/Macros.h
    Core/Log.h
    Core/Log.cpp
    Core/Paths.h
    Core/Paths.cpp
    Core/IConsoleManager.h
    Core/ConsoleManager.h
    Core/ConsoleManager.cpp
    Core/CommandLineOptions.h
    Core/CommandLineOptions.cpp
)

set(UTILITY_FILES
    Utilities/EnumUtil.h
    Utilities/Random.h
    Utilities/Delegate.h
    Utilities/Singleton.h
    Utilities/ThreadPool.h
    Utilities/ConcurrentQueue.h
    Utilities/StringUtil.h
    Utilities/StringUtil.cpp
    Utilities/ImageUtil.h
    Utilities/ImageUtil.cpp
    Utilities/FilesUtil.h
    Utilities/FilesUtil.cpp
    Utilities/JsonUtil.h
    Utilities/CpuBuffer2D.h
    Utilities/CLIParser.h
    Utilities/CLIParser.cpp
    Utilities/SDLUtil.h
    Utilities/SDLUtil.cpp
)


set(SCENE_FILES
    Scene/Scene.h
    Scene/Scene.cpp
    Scene/Camera.h
    Scene/Camera.cpp
    Scene/Light.h
    Scene/Mesh.h
    Scene/Material.h
)

set(EDITOR_FILES
    Editor/EditorConsole.h
    Editor/EditorConsole.cpp
    Editor/EditorSink.h
    Editor/EditorSink.cpp
    Editor/Editor.h
    Editor/Editor.cpp
)

set(EXTERNAL_COMMON_FILES
    ${CMAKE_SOURCE_DIR}/External/stb/stb_image.h
    ${CMAKE_SOURCE_DIR}/External/stb/stb_image_resize.h
    ${CMAKE_SOURCE_DIR}/External/stb/stb_image_write.h

    ${CMAKE_SOURCE_DIR}/External/ImGui/imconfig.h
    ${CMAKE_SOURCE_DIR}/External/ImGui/imgui.h
    ${CMAKE_SOURCE_DIR}/External/ImGui/imgui.cpp
    ${CMAKE_SOURCE_DIR}/External/ImGui/imgui_demo.cpp
    ${CMAKE_SOURCE_DIR}/External/ImGui/imgui_draw.cpp
    ${CMAKE_SOURCE_DIR}/External/ImGui/imgui_internal.h
    ${CMAKE_SOURCE_DIR}/External/ImGui/imgui_tables.cpp
    ${CMAKE_SOURCE_DIR}/External/ImGui/imgui_widgets.cpp
    ${CMAKE_SOURCE_DIR}/External/ImGui/imstb_rectpack.h
    ${CMAKE_SOURCE_DIR}/External/ImGui/imstb_textedit.h
    ${CMAKE_SOURCE_DIR}/External/ImGui/imstb_truetype.h

    ${CMAKE_SOURCE_DIR}/External/FontAwesome/IconsFontAwesome6.h
    ${CMAKE_SOURCE_DIR}/External/json/json.hpp

    ${CMAKE_SOURCE_DIR}/External/tinyobjloader/tiny_obj_loader.h
    ${CMAKE_SOURCE_DIR}/External/tinyobjloader/tiny_obj_loader.cc

    ${CMAKE_SOURCE_DIR}/External/cgltf/cgltf.h
)

set(EXTERNAL_FILES
    ${EXTERNAL_COMMON_FILES}
    ${CMAKE_SOURCE_DIR}/External/ImGui/imgui_impl_sdl2.h
    ${CMAKE_SOURCE_DIR}/External/ImGui/imgui_impl_sdl2.cpp
    ${CMAKE_SOURCE_DIR}/External/ImGui/imgui_impl_sdlrenderer2.h
    ${CMAKE_SOURCE_DIR}/External/ImGui/imgui_impl_sdlrenderer2.cpp
)

if(APPLE)
    set(DEVICE_BACKEND_FILES
        Device/PathTracer.h
        Device/Metal/MetalPathTracer.h
        Device/Metal/MetalPathTracer.cpp
    )
else()
    set(DEVICE_BACKEND_FILES
        Device/PathTracer.h
        Device/OptiX/OptixUtils.h
        Device/OptiX/OptixUtils.cpp
        Device/OptiX/OptixPathTracer.h
        Device/OptiX/OptixPathTracer.cpp
        Device/OptiX/DeviceHostCommon.h
        Device/OptiX/KernelCompiler.h
        Device/OptiX/KernelCompiler.cpp
    )
endif()

set(PLATFORM_FILES
    Platform/Window.h
    Platform/Window.cpp
    Platform/Input.cpp
    Platform/Input.h
)

if(NOT APPLE)
    set(CUDA_KERNEL_FILES
        Device/OptiX/Kernels/PostProcessing.cu
        Device/OptiX/Kernels/PRNG.cuh
        Device/OptiX/Kernels/Color.cuh
        Device/OptiX/Kernels/ONB.cuh
        Device/OptiX/Kernels/Material.cuh
        Device/OptiX/Kernels/Disney.cuh
        Device/OptiX/Kernels/Math.cuh
        Device/OptiX/Kernels/DeviceCommon.cuh
        Device/OptiX/Kernels/DeviceTypes.cuh
        Device/OptiX/Kernels/Impl/MathImpl.cuh
        Device/OptiX/Kernels/Impl/MathDefines.cuh
    )

    set(OPTIX_KERNEL_FILES
        Device/OptiX/Kernels/PathTracing.cu
    )
else()
    set(CUDA_KERNEL_FILES "")
    set(OPTIX_KERNEL_FILES "")
endif()

set(MATH_FILES
    Math/MathTypes.h
)

set(MAIN_FILES
    main.cpp
)

set(MISC_FILES
    Saved/Scenes/sponza.json
    Saved/Scenes/sanmiguel.json
    Saved/Scenes/toyshop.json
    Saved/Scenes/salle_de_bain.json
    Saved/Scenes/living_room.json

    Saved/Ini/cvars.ini
)

set_source_files_properties(${MISC_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)


add_executable(Amber ${MAIN_FILES} ${MATH_FILES} ${DEVICE_BACKEND_FILES} ${OPTIX_KERNEL_FILES} ${CUDA_KERNEL_FILES} ${CORE_FILES} ${UTILITY_FILES} ${RENDERER_FILES} ${SCENE_FILES} ${EDITOR_FILES} ${EXTERNAL_FILES} ${PLATFORM_FILES} ${MISC_FILES})
set_target_properties(Amber PROPERTIES OUTPUT_NAME Amber)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
# Group source files within Amber/ directory
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${MAIN_FILES} ${MATH_FILES} ${DEVICE_BACKEND_FILES} ${OPTIX_KERNEL_FILES} ${CUDA_KERNEL_FILES} ${CORE_FILES} ${UTILITY_FILES} ${RENDERER_FILES} ${SCENE_FILES} ${EDITOR_FILES} ${PLATFORM_FILES} ${MISC_FILES})
# Group external files separately (they're in root/External/)
source_group(TREE ${CMAKE_SOURCE_DIR}/External PREFIX "External" FILES ${EXTERNAL_FILES})

if(NOT APPLE)
    foreach(file ${OPTIX_KERNEL_FILES})
        set_source_files_properties(${file} PROPERTIES HEADER_FILE_ONLY TRUE)
    endforeach()
endif()

target_include_directories(Amber PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Amber PRIVATE ${CMAKE_SOURCE_DIR}/External)
target_include_directories(Amber PRIVATE ${CMAKE_SOURCE_DIR}/External/spdlog/include)
target_include_directories(Amber PRIVATE ${CMAKE_SOURCE_DIR}/External/pbrtParser)

if(APPLE)
    # SDL2 headers are in External/SDL2/include/SDL2/, so add parent directory
    target_include_directories(Amber PRIVATE ${CMAKE_SOURCE_DIR}/External/SDL2/include)

    target_compile_definitions(Amber PRIVATE SDL_VIDEO_DRIVER_COCOA)

    # Use bundled SDL2.framework
    set(SDL2_FRAMEWORK_PATH "${CMAKE_SOURCE_DIR}/External/SDL2/lib/SDL2.framework")
    if(EXISTS ${SDL2_FRAMEWORK_PATH})
        target_link_libraries(Amber PRIVATE ${SDL2_FRAMEWORK_PATH})
        message(STATUS "Using bundled SDL2.framework from External/SDL2/lib")
    else()
        # Fallback: try to find system SDL2
        find_package(SDL2 QUIET)
        if(SDL2_FOUND)
            target_link_libraries(Amber PRIVATE SDL2::SDL2)
            message(STATUS "Using system SDL2")
        else()
            message(FATAL_ERROR "SDL2.framework not found at ${SDL2_FRAMEWORK_PATH}. Please add SDL2.framework to External/SDL2/lib or install via 'brew install sdl2'")
        endif()
    endif()
else()
    target_include_directories(Amber PRIVATE ${CUDA_INCLUDE_DIRS})
    target_include_directories(Amber PRIVATE ${SDL2_INCLUDE_DIRS})
    target_include_directories(Amber PRIVATE ${CMAKE_SOURCE_DIR}/External/SDL2/include)
    target_include_directories(Amber PRIVATE ${OPTIX_INSTALL_DIR}/include)

    target_link_libraries(Amber PUBLIC  ${CUDA_LIBRARIES})
    target_link_libraries(Amber PUBLIC  ${NVRTC_LIB})
    target_link_libraries(Amber PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_link_libraries(Amber PRIVATE ${CMAKE_SOURCE_DIR}/External/SDL2/lib/SDL2.lib)
endif()

target_link_libraries(Amber PRIVATE pbrtParser)

if(MSVC)
    set(DISABLED_WARNINGS /wd4996)
    target_compile_options(Amber
        PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            ${DISABLED_WARNINGS}
            /Zc:preprocessor
            /FI"Core/Types.h"
            /FI"Core/Macros.h"
            /FI"Math/MathTypes.h"
        >
    )
elseif(APPLE OR UNIX)
    set(ALL_SOURCE_FILES
        ${MAIN_FILES}
        ${CORE_FILES}
        ${UTILITY_FILES}
        ${SCENE_FILES}
        ${EDITOR_FILES}
        ${PLATFORM_FILES}
        ${DEVICE_BACKEND_FILES}
    )
    set_source_files_properties(${ALL_SOURCE_FILES}
        PROPERTIES
        COMPILE_FLAGS "-include ${CMAKE_CURRENT_SOURCE_DIR}/Core/Types.h -include ${CMAKE_CURRENT_SOURCE_DIR}/Core/Macros.h -include ${CMAKE_CURRENT_SOURCE_DIR}/Math/MathTypes.h"
    )
endif()

set(AMBER_PATH ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(Amber PRIVATE AMBER_PATH="${AMBER_PATH}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(Amber PRIVATE AMBER_DEBUG=1)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(Amber PRIVATE AMBER_RELEASE=1)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_definitions(Amber PRIVATE AMBER_RELWITHDEBINFO=1)
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    target_compile_definitions(Amber PRIVATE AMBER_MINSIZEREL=1)
endif()

if(NOT APPLE)
    target_compile_definitions(Amber PRIVATE CUDA_PATH="${CUDA_TOOLKIT_ROOT_DIR}")
    target_compile_definitions(Amber PRIVATE OPTIX_PATH="${OPTIX_INSTALL_DIR}")
endif()

if(WIN32)
    target_compile_definitions(Amber PRIVATE NOMINMAX)
endif()

install(TARGETS Amber DESTINATION bin)

if(WIN32)
    if(CMAKE_CONFIGURATION_TYPES)
        set(SDL2_DLL_NAME "$<$<CONFIG:Debug>:SDL2d.dll>$<$<CONFIG:Release>:SDL2.dll>")
    else()
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(SDL2_DLL_NAME "SDL2d.dll")
        else()
            set(SDL2_DLL_NAME "SDL2.dll")
        endif()
    endif()

    add_custom_command(TARGET Amber PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/External/SDL2/lib/SDL2.dll"
            "$<TARGET_FILE_DIR:Amber>"
    )
endif()