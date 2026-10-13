enable_testing()

set(TEST_FIXTURES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

macro(add_arda_unit_test name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE arda_shared)
    target_compile_definitions(${name} PRIVATE FIXTURES_DIR="${TEST_FIXTURES_DIR}")
    add_test(NAME ${name} COMMAND ${name})
endmacro()

add_arda_unit_test(test_cpu_detection unit/test_cpu_detection.c)
add_arda_unit_test(test_memory_detection unit/test_memory_detection.c)
add_arda_unit_test(test_profile_selection unit/test_profile_selection.c)
add_arda_unit_test(test_capability_format unit/test_capability_format.c)
add_arda_unit_test(test_live_probe unit/test_live_probe.c)
