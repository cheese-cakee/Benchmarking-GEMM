cmake_minimum_required(VERSION 3.16)
project(gemm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(GEMM_NATIVE "Tune for the build machine (-march=native); otherwise target AVX2 + FMA" ON)
option(GEMM_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

find_package(OpenMP REQUIRED)

if(GEMM_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
    add_link_options(-fsanitize=address,undefined)
endif()

if(GEMM_NATIVE)
    set(GEMM_ARCH_FLAGS -march=native)
else()
    set(GEMM_ARCH_FLAGS -mavx2 -mfma)
endif()

add_library(gemm
    src/kernels/scalar.cpp
    src/kernels/tiled.cpp
    src/kernels/avx2.cpp
    src/kernels/packed.cpp
    src/registry.cpp
)
target_include_directories(gemm PUBLIC include)
target_compile_options(gemm PRIVATE -Wall -Wextra -ffast-math PUBLIC ${GEMM_ARCH_FLAGS})
target_link_libraries(gemm PUBLIC OpenMP::OpenMP_CXX)

add_executable(gemm_bench bench/gemm_bench.cpp)
target_link_libraries(gemm_bench PRIVATE gemm)
target_compile_options(gemm_bench PRIVATE -Wall -Wextra)
string(REPLACE ";" " " GEMM_ARCH_TEXT "${GEMM_ARCH_FLAGS}")
target_compile_definitions(gemm_bench PRIVATE
    GEMM_BUILD_DESCRIPTION="${CMAKE_BUILD_TYPE} ${GEMM_ARCH_TEXT} -ffast-math -fopenmp")

enable_testing()
add_executable(gemm_tests tests/test_correctness.cpp)
target_link_libraries(gemm_tests PRIVATE gemm)
target_compile_options(gemm_tests PRIVATE -Wall -Wextra)
add_test(NAME correctness COMMAND gemm_tests)
add_test(NAME bench_smoke COMMAND gemm_bench --sizes 96 --warmup 0 --reps 1)
