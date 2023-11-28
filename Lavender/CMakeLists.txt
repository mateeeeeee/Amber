

#set(CORE_FILES
#  Core/CoreTypes.h
#  Core/Defines.h
#  Core/Logger.h
#)

add_executable(Lavender main.cpp)

set_target_properties(Lavender PROPERTIES OUTPUT_NAME Lavender)

source_group(" " 		FILES main.cpp)

target_include_directories(Lavender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
#target_compile_definitions(Lavender PRIVATE ${LLVM_DEFINITIONS})

#set(LIBRARIES
#
#)
#
#target_link_directories(Lavender PRIVATE ${})
#target_link_libraries(Lavender PRIVATE ${LIBRARIES})

#set(DISABLED_WARNINGS
#	
#)
#
#target_compile_options(Lavender PRIVATE ${DISABLED_WARNINGS})
#
#if(MSVC)
#    add_definitions(/FI"Core/CoreTypes.h" /FI"Core/Defines.h")
#else()                                                   
#    add_definitions(-include Core/CoreTypes.h -include Core/Defines.h)
#endif()

install(TARGETS Lavender DESTINATION bin)
