
set(TARGET_NAME Simple2DRendererViewer)
file(
        GLOB_RECURSE SRCS
        "${PROJECT_SOURCE_DIR}/gui/*.cpp"
        "${PROJECT_SOURCE_DIR}/gui/*.hpp"
)


add_executable(${TARGET_NAME} ${SRCS})

target_link_libraries(${TARGET_NAME} PRIVATE CGUtils)

target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)