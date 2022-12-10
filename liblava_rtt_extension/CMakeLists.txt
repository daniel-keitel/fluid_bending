cmake_minimum_required(VERSION 3.23)

set(LIBLAVA_EXTENSION_DIR ${CMAKE_CURRENT_SOURCE_DIR})

message(">> lava::rtt_extension")

FILE(GLOB LIBLAVA_EXTENSION_SOURCES ${LIBLAVA_EXTENSION_DIR}/src/*.cpp)

add_library(lava.rtt_extension STATIC ${LIBLAVA_EXTENSION_SOURCES} )

target_link_libraries(lava.rtt_extension PUBLIC
        lava::engine
        )

target_include_directories(lava.rtt_extension INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        )

set_property(TARGET lava.rtt_extension PROPERTY EXPORT_NAME rtt_extension)
add_library(lava::rtt_extension ALIAS lava.rtt_extension)