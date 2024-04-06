
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(OPTIX_INSTALL_DIR "C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 8.0.0")
#dont hardcode values


set(CORE_FILES
    Core/CoreTypes.h
    Core/Defines.h
    Core/Logger.h
    Core/Logger.cpp
    Core/Window.h
    Core/Window.cpp
    Core/Input.h
    Core/Input.cpp
    Core/Paths.h
    Core/Paths.cpp

    Core/ConsoleManager.h
    Core/ConsoleManager.cpp
    Core/ConsoleVariable.h
    Core/ConsoleVariable.cpp
    Core/ConsoleCommand.h
    Core/ConsoleCommand.cpp
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

set(RENDERER_FILES
)

set(SCENE_FILES
    Scene/Scene.h
    Scene/Scene.cpp
    Scene/Camera.h
    Scene/Light.h
    Scene/Mesh.h
    Scene/Material.h
    Scene/Material.cpp
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
)

set(OPTIX_BACKEND_FILES
    Optix/MathUtils.h
    Optix/CudaUtils.h
    Optix/CudaUtils.cpp
    Optix/OptixUtils.h
    Optix/OptixUtils.cpp
    Optix/OptixRenderer.h
    Optix/OptixRenderer.cpp
    Optix/OptixShared.h
)

set(MATH_FILES
    Math/MathTypes.h
)

set(MAIN_FILES
    main.cpp
)


add_executable(Lavender ${MAIN_FILES} ${MATH_FILES} ${OPTIX_BACKEND_FILES} ${CORE_FILES} ${UTILITY_FILES} ${RENDERER_FILES} ${SCENE_FILES} ${EDITOR_FILES} ${EXTERNAL_FILES})
set_target_properties(Lavender PROPERTIES OUTPUT_NAME Lavender)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${MAIN_FILES} ${MATH_FILES} ${OPTIX_BACKEND_FILES} "Optix/OptixRenderer.cu" ${CORE_FILES} ${UTILITY_FILES} ${RENDERER_FILES} ${SCENE_FILES} ${EDITOR_FILES} ${EXTERNAL_FILES})

target_include_directories(Lavender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Lavender PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(Lavender PRIVATE ${SDL2_INCLUDE_DIRS})
target_include_directories(Lavender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/External)
target_include_directories(Lavender PRIVATE ${CMAKE_SOURCE_DIR}/Libraries/spdlog/include)
target_include_directories(Lavender PRIVATE ${CMAKE_SOURCE_DIR}/Libraries/DirectXMath/include)
target_include_directories(Lavender PRIVATE ${CMAKE_SOURCE_DIR}/Libraries/CLI11/include)
target_include_directories(Lavender PRIVATE ${CMAKE_SOURCE_DIR}/Libraries/pbrtParser)
target_include_directories(Lavender PRIVATE ${CMAKE_SOURCE_DIR}/Libraries/SDL2/include)
target_include_directories(Lavender PRIVATE ${OPTIX_INSTALL_DIR}/include)


target_link_libraries(Lavender PUBLIC  ${CUDA_LIBRARIES})
target_link_libraries(Lavender PRIVATE CUDA::cudart CUDA::cuda_driver)
target_link_libraries(Lavender PRIVATE ${CMAKE_SOURCE_DIR}/Libraries/SDL2/lib/SDL2.lib)
target_link_libraries(Lavender PRIVATE pbrtParser)

set(DISABLED_WARNINGS
	/wd4996
)

if(MSVC)
    target_compile_options(Lavender
        PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            ${DISABLED_WARNINGS}
            /FI"Core/CoreTypes.h"
            /FI"Core/Defines.h"
            /FI"Math/MathTypes.h"
        >
    )
else()
    target_compile_options(Lavender
        PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            ${DISABLED_WARNINGS}
            -include Core/CoreTypes.h
            -include Core/Defines.h
            -include Math/MathTypes.h
        >
    )
endif()

target_compile_options(Lavender
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --pre-include=${CMAKE_CURRENT_SOURCE_DIR}/Core/CoreTypes.h
        --pre-include=${CMAKE_CURRENT_SOURCE_DIR}/Core/Defines.h
        --pre-include=${CMAKE_CURRENT_SOURCE_DIR}/Math/MathTypes.h
    >
)


set(LAVENDER_PATH ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(Lavender PRIVATE LAVENDER_PATH="${LAVENDER_PATH}")

if(WIN32)
    target_compile_definitions(Lavender PRIVATE NOMINMAX)
endif()


install(TARGETS Lavender DESTINATION bin)


if(CMAKE_CONFIGURATION_TYPES)
    set(SDL2_DLL_NAME "$<$<CONFIG:Debug>:SDL2d.dll>$<$<CONFIG:Release>:SDL2.dll>")
else()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(SDL2_DLL_NAME "SDL2d.dll")
    else()
        set(SDL2_DLL_NAME "SDL2.dll")
    endif()
endif()

add_custom_command(TARGET Lavender PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/Libraries/SDL2/lib/SDL2.dll"
        "$<TARGET_FILE_DIR:Lavender>"
)


set(OPTIX_SHADER_FILES
    Optix/OptixRenderer.cu
)

add_library(PTX OBJECT ${OPTIX_SHADER_FILES})
set_property(TARGET PTX PROPERTY CUDA_PTX_COMPILATION ON)
target_include_directories(PTX PRIVATE ${OPTIX_INSTALL_DIR}/include)
target_include_directories(PTX PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(PTX PRIVATE ${CUDA_INCLUDE_DIRS})

target_compile_options(PTX PRIVATE "--device-debug" "-O0" "--generate-line-info")

set(PTX_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ptx")

add_dependencies(Lavender PTX)
