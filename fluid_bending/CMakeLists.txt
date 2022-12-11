cmake_minimum_required(VERSION 3.23)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} /MANIFEST:NO")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /MANIFEST:NO")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /MANIFEST:NO")
endif()

set(ENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

message(">> engine")

FILE(GLOB ENGINE_SOURCES ${ENGINE_DIR}/src/*.cpp)

add_executable(fluid_bending ${ENGINE_SOURCES})

target_link_libraries(fluid_bending lava::engine
                             shaderc
                             assimp
                             glfw
                             lava::rtt_extension)