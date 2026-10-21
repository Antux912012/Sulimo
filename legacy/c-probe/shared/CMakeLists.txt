add_library(arda_shared STATIC
    src/capability_os.c
    src/capability_cpu.c
    src/capability_memory.c
    src/capability_gpu.c
    src/capability_report.c
    src/profile.c
)

target_include_directories(arda_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(arda_shared PRIVATE
    ${CMAKE_DL_LIBS}
)

if(ARDA_ENABLE_VULKAN)
    target_compile_definitions(arda_shared PRIVATE ARDA_BACKEND_VULKAN_ENABLED)
endif()

if(ARDA_ENABLE_OPENCL)
    target_compile_definitions(arda_shared PRIVATE ARDA_BACKEND_OPENCL_ENABLED)
endif()

if(ARDA_ENABLE_SYCL)
    target_compile_definitions(arda_shared PRIVATE ARDA_BACKEND_SYCL_ENABLED)
endif()
