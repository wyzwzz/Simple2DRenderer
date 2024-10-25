
set(TARGET_NAME Simple2DRenderer_Core)
file(
        GLOB_RECURSE SRCS
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/*.hpp"
)


add_library(${TARGET_NAME} ${SRCS})

target_link_libraries(${TARGET_NAME} PRIVATE CGUtils)

target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)