cmake_minimum_required(VERSION 3.17)

project(Simple2DRenderer LANGUAGES CXX C)

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set(BUILD_ENABLE_OPENGL ON)
add_subdirectory(third_party/CGUtils)

# add_subdirectory(src)

add_subdirectory(gui)



