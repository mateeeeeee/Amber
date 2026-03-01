add_executable(BVHBenchmark BVHBenchmark.cpp)
set_target_properties(BVHBenchmark PROPERTIES OUTPUT_NAME BVHBenchmark)
target_link_libraries(BVHBenchmark PRIVATE AmberLib)

if(MSVC)
    # Force-includes propagate from AmberLib via PUBLIC
elseif(APPLE OR UNIX)
    set(AMBER_SRC ${CMAKE_SOURCE_DIR}/Amber)
    set_source_files_properties(BVHBenchmark.cpp
        PROPERTIES
        COMPILE_FLAGS "-include ${AMBER_SRC}/Core/Types.h -include ${AMBER_SRC}/Core/Macros.h -include ${AMBER_SRC}/Math/MathTypes.h"
    )
endif()
