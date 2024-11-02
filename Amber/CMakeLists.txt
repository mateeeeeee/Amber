
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(OPTIX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0")

find_library(NVRTC_LIB nvrtc PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

if(NOT NVRTC_LIB)
    message(FATAL_ERROR "NVRTC library not found!")
endif()


set(CORE_FILES
    Core/Types.h
    Core/Macros.h
    Core/Logger.h
    Core/Logger.cpp
    Core/Window.h
    Core/Window.cpp
    Core/Input.h
    Core/Input.cpp
    Core/Paths.h
    Core/Paths.cpp
    Core/IConsoleManager.h
    Core/ConsoleManager.h
    Core/ConsoleManager.cpp
)

set(UTILITY_FILES
    Utilities/EnumUtil.h
    Utilities/Random.h
    Utilities/Delegate.h
    Utilities/SDLUtil.h
    Utilities/SDLUtil.cpp
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
    Editor/Editor.h
    Editor/Editor.cpp
    Editor/EditorConsole.h
    Editor/EditorConsole.cpp
    Editor/EditorSink.h
    Editor/EditorSink.cpp
)

set(EXTERNAL_FILES
    External/SimpleMath/SimpleMath.h
    External/SimpleMath/SimpleMath.inl
    External/SimpleMath/SimpleMath.cpp

    External/stb/stb_image.h
    External/stb/stb_image_resize.h
    External/stb/stb_image_write.h

    External/ImGui/imconfig.h
    External/ImGui/imgui.h
    External/ImGui/imgui.cpp
    External/ImGui/imgui_demo.cpp
    External/ImGui/imgui_draw.cpp
    External/ImGui/imgui_impl_sdl.h
    External/ImGui/imgui_impl_sdl.cpp
    External/ImGui/imgui_impl_sdlrenderer.h
    External/ImGui/imgui_impl_sdlrenderer.cpp
    External/ImGui/imgui_internal.h
    External/ImGui/imgui_tables.cpp
    External/ImGui/imgui_widgets.cpp
    External/ImGui/imstb_rectpack.h
    External/ImGui/imstb_textedit.h
    External/ImGui/imstb_truetype.h

    External/FontAwesome/IconsFontAwesome6.h
    External/json/json.hpp

    External/tinyobjloader/tiny_obj_loader.h
    External/tinyobjloader/tiny_obj_loader.cc

    External/cgltf/cgltf.h
)

set(OPTIX_BACKEND_FILES
    Optix/OptixUtils.h
    Optix/OptixUtils.cpp
    Optix/OptixRenderer.h
    Optix/OptixRenderer.cpp
    Optix/OptixShared.h
    Optix/KernelCompiler.h
    Optix/KernelCompiler.cpp

)

 set(OPTIX_KERNEL_FILES
    Optix/Kernels/Math.cuh
    Optix/Kernels/Random.cuh
    Optix/Kernels/Color.cuh
    Optix/Kernels/Sampling.cuh
    Optix/Kernels/Disney.cuh
    Optix/Kernels/PathTracing.cu
 )

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


add_executable(Amber ${MAIN_FILES} ${MATH_FILES} ${OPTIX_BACKEND_FILES} ${OPTIX_KERNEL_FILES} ${CORE_FILES} ${UTILITY_FILES} ${RENDERER_FILES} ${SCENE_FILES} ${EDITOR_FILES} ${EXTERNAL_FILES} ${MISC_FILES})
set_target_properties(Amber PROPERTIES OUTPUT_NAME Amber)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${MAIN_FILES} ${MATH_FILES} ${OPTIX_BACKEND_FILES} ${OPTIX_KERNEL_FILES} ${CORE_FILES} ${UTILITY_FILES} ${RENDERER_FILES} ${SCENE_FILES} ${EDITOR_FILES} ${EXTERNAL_FILES} ${MISC_FILES})

# Exclude the .cu files from the build
foreach(file ${OPTIX_KERNEL_FILES})
    set_source_files_properties(${file} PROPERTIES HEADER_FILE_ONLY TRUE)
endforeach()

target_include_directories(Amber PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Amber PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(Amber PRIVATE ${SDL2_INCLUDE_DIRS})
target_include_directories(Amber PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/External)
target_include_directories(Amber PRIVATE ${CMAKE_SOURCE_DIR}/Libraries/spdlog/include)
target_include_directories(Amber PRIVATE ${CMAKE_SOURCE_DIR}/Libraries/DirectXMath/include)
target_include_directories(Amber PRIVATE ${CMAKE_SOURCE_DIR}/Libraries/CLI11/include)
target_include_directories(Amber PRIVATE ${CMAKE_SOURCE_DIR}/Libraries/pbrtParser)
target_include_directories(Amber PRIVATE ${CMAKE_SOURCE_DIR}/Libraries/SDL2/include)
target_include_directories(Amber PRIVATE ${OPTIX_INSTALL_DIR}/include)


target_link_libraries(Amber PUBLIC  ${CUDA_LIBRARIES})
target_link_libraries(Amber PUBLIC  ${NVRTC_LIB})
target_link_libraries(Amber PRIVATE CUDA::cudart CUDA::cuda_driver)
target_link_libraries(Amber PRIVATE ${CMAKE_SOURCE_DIR}/Libraries/SDL2/lib/SDL2.lib)
target_link_libraries(Amber PRIVATE pbrtParser)

set(DISABLED_WARNINGS
	/wd4996
)

if(MSVC)
    target_compile_options(Amber
        PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            ${DISABLED_WARNINGS}
            /FI"Core/Types.h"
            /FI"Core/Macros.h"
            /FI"Math/MathTypes.h"
        >
    )
else()
    target_compile_options(Amber
        PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            ${DISABLED_WARNINGS}
            -include Core/Types.h
            -include Core/Macros.h
            -include Math/MathTypes.h
        >
    )
endif()


set(AMBER_PATH ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(Amber PRIVATE AMBER_PATH="${AMBER_PATH}")
target_compile_definitions(Amber PRIVATE CUDA_PATH="${CUDA_TOOLKIT_ROOT_DIR}")
target_compile_definitions(Amber PRIVATE OPTIX_PATH="${OPTIX_INSTALL_DIR}")

if(WIN32)
    target_compile_definitions(Amber PRIVATE NOMINMAX)
endif()


install(TARGETS Amber DESTINATION bin)


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
        "${CMAKE_SOURCE_DIR}/Libraries/SDL2/lib/SDL2.dll"
        "$<TARGET_FILE_DIR:Amber>"
)
