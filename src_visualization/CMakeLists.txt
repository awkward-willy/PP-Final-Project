cmake_minimum_required(VERSION 3.16)
project(AntVisualization CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenMP REQUIRED)
find_package(SDL2 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2_TTF REQUIRED SDL2_ttf)

# Add executable
add_executable(ant_visualization main.cpp)

# Include directories
target_include_directories(ant_visualization PRIVATE
    ${SDL2_INCLUDE_DIRS}
    ${SDL2_TTF_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(ant_visualization PRIVATE
    OpenMP::OpenMP_CXX
    ${SDL2_LIBRARIES}
    ${SDL2_TTF_LIBRARIES}
)

# Compiler flags
target_compile_options(ant_visualization PRIVATE -O3 -march=native)
