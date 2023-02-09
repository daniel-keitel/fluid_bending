cmake_minimum_required(VERSION 3.22)

project(fluid_bending)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
add_compile_options("/Zc:__cplusplus")
endif()

add_subdirectory(ext)

include(cmake/CPM.cmake)

CPMAddPackage("gh:assimp/assimp@5.2.5")

CPMAddPackage(
        NAME glm
        GITHUB_REPOSITORY ${glm_GITHUB}
        GIT_TAG ${glm_TAG}
        DOWNLOAD_ONLY YES
)

add_subdirectory(liblava_rtt_extension)
add_subdirectory(fluid_bending)
