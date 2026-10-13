cmake_minimum_required(VERSION 3.20)
project(arda_breath VERSION 0.1.0 LANGUAGES C)

# Modern C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Build switches
option(ARDA_BUILD_TESTS "Build unit tests" ON)
option(ARDA_BUILD_PROBE "Build arda-probe CLI tool" ON)

# Independent backend toggles
option(ARDA_ENABLE_VULKAN "Enable Vulkan backend integration" OFF)
option(ARDA_ENABLE_OPENCL "Enable OpenCL backend integration" OFF)
option(ARDA_ENABLE_SYCL   "Enable Intel SYCL backend integration" OFF)

# Optimization target policy (default: portable baseline)
set(ARDA_CPU_TARGET "portable" CACHE STRING "CPU optimization profile: portable, classic, current, native")
set_property(CACHE ARDA_CPU_TARGET PROPERTY STRINGS "portable" "classic" "current" "native")

# High-quality compiler warnings
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wformat=2
        -Wconversion
        -Wsign-conversion
        -Wno-unused-parameter
    )
endif()

# Report configuration summary
message(STATUS "--------------------------------------------------------")
message(STATUS "Arda's Breath - Configuration Summary")
message(STATUS "  Version:            ${PROJECT_VERSION}")
message(STATUS "  Build Type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "  CPU Target Policy:  ${ARDA_CPU_TARGET}")
message(STATUS "  Vulkan Backend:     ${ARDA_ENABLE_VULKAN}")
message(STATUS "  OpenCL Backend:     ${ARDA_ENABLE_OPENCL}")
message(STATUS "  Intel SYCL Backend: ${ARDA_ENABLE_SYCL}")
message(STATUS "  Build Tests:        ${ARDA_BUILD_TESTS}")
message(STATUS "  Build Probe CLI:    ${ARDA_BUILD_PROBE}")
message(STATUS "--------------------------------------------------------")

# Shared capability probe and runtime policy library
add_subdirectory(shared)

# Command-line probe utility
if(ARDA_BUILD_PROBE)
    add_subdirectory(tools/arda-probe)
endif()

# Unit tests and test fixtures
if(ARDA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
