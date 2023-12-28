

find_package(SDL2 REQUIRED)
find_package(CUDAToolkit)

set(CORE_FILES
    Core/CoreTypes.h
    Core/Defines.h
    Core/Logger.h
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
)

set(RENDERER_FILES
    Renderer/Renderer.h
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
)

set(CUDA_FILES
    Cuda/CudaUtil.h
    Cuda/CudaUtil.cpp
    Cuda/CudaEvent.h
    Cuda/CudaEvent.cpp
    Cuda/CudaStream.h
    Cuda/CudaStream.cpp
    Cuda/CudaAlloc.h
    Cuda/CudaAlloc.cpp
    Cuda/CudaKernel.h
)

set(MAIN_FILES
    main.cpp
)

add_executable(Lavender ${MAIN_FILES} ${CUDA_FILES} ${CORE_FILES} ${UTILITY_FILES} ${RENDERER_FILES} ${EDITOR_FILES} ${EXTERNAL_FILES})
set_target_properties(Lavender PROPERTIES OUTPUT_NAME Lavender)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES  ${MAIN_FILES} ${CUDA_FILES} ${CORE_FILES} ${UTILITY_FILES} ${RENDERER_FILES} ${EDITOR_FILES} ${EXTERNAL_FILES})

target_include_directories(Lavender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Lavender PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(Lavender PRIVATE ${SDL2_INCLUDE_DIRS})
target_include_directories(Lavender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/External)
target_include_directories(Lavender PRIVATE ${CMAKE_SOURCE_DIR}/Libraries/spdlog/include)
target_include_directories(Lavender PRIVATE ${CMAKE_SOURCE_DIR}/Libraries/DirectXMath/include)
target_include_directories(Lavender PRIVATE ${CMAKE_SOURCE_DIR}/Libraries/CLI11/include)
target_include_directories(Lavender PRIVATE ${CMAKE_SOURCE_DIR}/Libraries/pbrtParser)


target_link_libraries(Lavender PUBLIC  ${CUDA_LIBRARIES})
target_link_libraries(Lavender PRIVATE CUDA::cudart CUDA::cuda_driver)
target_link_libraries(Lavender PRIVATE ${SDL2_LIBRARIES})
target_link_libraries(Lavender PRIVATE pbrtParser)

if(MSVC)
    add_definitions(/FI"Core/CoreTypes.h" /FI"Core/Defines.h")
else()                                                   
    add_definitions(-include Core/CoreTypes.h -include Core/Defines.h)
endif()

set(LAVENDER_PATH ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(Lavender PRIVATE LAVENDER_PATH="${LAVENDER_PATH}")


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

add_custom_command(TARGET Lavender POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SDL2_DIR}/../bin/${SDL2_DLL_NAME}"
        "$<TARGET_FILE_DIR:Lavender>"
)

