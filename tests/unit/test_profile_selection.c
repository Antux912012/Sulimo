#include "arda/capability.h"
#include "arda/profile.h"
#include "../test_helpers.h"

static void test_classic_profile_selection(void) {
    arda_cpu_info_t cpu;
    memset(&cpu, 0, sizeof(cpu));
    cpu.has_avx = true;
    cpu.has_avx2 = true;
    cpu.has_fma = true;
    cpu.has_avx512f = false;
    cpu.has_avx_vnni = false;

    arda_memory_info_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.total_ram_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL; /* 16 GB */

    arda_backend_status_t backends[4];
    memset(backends, 0, sizeof(backends));
    backends[0].id = ARDA_BACKEND_CPU;
    backends[0].runtime_available = true;

    arda_profile_info_t prof;
    arda_profile_evaluate(&cpu, &mem, backends, 1, &prof);

    ASSERT_EQ_INT(prof.type, ARDA_PROFILE_CLASSIC);
    ASSERT_STR_CONTAINS(prof.name, "arda-classic");
    ASSERT_STR_CONTAINS(prof.reason, "AVX2");
    ASSERT_STR_CONTAINS(prof.reason, "Classic profile selected");
}

static void test_portable_profile_selection(void) {
    arda_cpu_info_t cpu;
    memset(&cpu, 0, sizeof(cpu));
    cpu.has_sse4_2 = true;
    cpu.has_avx = false;
    cpu.has_avx2 = false;

    arda_memory_info_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.total_ram_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL; /* 8 GB */

    arda_backend_status_t backends[1];
    memset(backends, 0, sizeof(backends));
    backends[0].id = ARDA_BACKEND_CPU;
    backends[0].runtime_available = true;

    arda_profile_info_t prof;
    arda_profile_evaluate(&cpu, &mem, backends, 1, &prof);

    ASSERT_EQ_INT(prof.type, ARDA_PROFILE_PORTABLE);
    ASSERT_STR_CONTAINS(prof.name, "arda-portable");
    ASSERT_STR_CONTAINS(prof.reason, "lacks AVX2");
}

static void test_current_profile_selection_avx512(void) {
    arda_cpu_info_t cpu;
    memset(&cpu, 0, sizeof(cpu));
    cpu.has_avx2 = true;
    cpu.has_avx512f = true;
    cpu.has_avx512_vnni = true;

    arda_memory_info_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.total_ram_bytes = 32ULL * 1024ULL * 1024ULL * 1024ULL; /* 32 GB */

    arda_backend_status_t backends[1];
    memset(backends, 0, sizeof(backends));
    backends[0].id = ARDA_BACKEND_CPU;
    backends[0].runtime_available = true;

    arda_profile_info_t prof;
    arda_profile_evaluate(&cpu, &mem, backends, 1, &prof);

    ASSERT_EQ_INT(prof.type, ARDA_PROFILE_CURRENT);
    ASSERT_STR_CONTAINS(prof.name, "arda-current");
    ASSERT_STR_CONTAINS(prof.reason, "AVX-512 VNNI");
}

static void test_current_profile_selection_avx_vnni(void) {
    arda_cpu_info_t cpu;
    memset(&cpu, 0, sizeof(cpu));
    cpu.has_avx2 = true;
    cpu.has_avx_vnni = true;

    arda_memory_info_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.total_ram_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL; /* 16 GB */

    arda_backend_status_t backends[1];
    memset(backends, 0, sizeof(backends));
    backends[0].id = ARDA_BACKEND_CPU;
    backends[0].runtime_available = true;

    arda_profile_info_t prof;
    arda_profile_evaluate(&cpu, &mem, backends, 1, &prof);

    ASSERT_EQ_INT(prof.type, ARDA_PROFILE_CURRENT);
    ASSERT_STR_CONTAINS(prof.name, "arda-current");
    ASSERT_STR_CONTAINS(prof.reason, "AVX-VNNI");
}

static void test_null_handling(void) {
    arda_profile_info_t prof;
    arda_profile_evaluate(NULL, NULL, NULL, 0, &prof);

    ASSERT_EQ_INT(prof.type, ARDA_PROFILE_PORTABLE);
    ASSERT_STR_CONTAINS(prof.name, "arda-portable");
    ASSERT_STR_CONTAINS(prof.reason, "Missing CPU or memory");
}

int main(void) {
    printf("[Unit Test] Profile Evaluation & Policy Rules\n");
    RUN_TEST(test_classic_profile_selection);
    RUN_TEST(test_portable_profile_selection);
    RUN_TEST(test_current_profile_selection_avx512);
    RUN_TEST(test_current_profile_selection_avx_vnni);
    RUN_TEST(test_null_handling);
    TEST_REPORT_MAIN();
}
