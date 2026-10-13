#include "arda/capability.h"
#include "../test_helpers.h"

#ifndef FIXTURES_DIR
#define FIXTURES_DIR "fixtures"
#endif

static void test_kaby_lake_fixture(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/cpuinfo/i5_7200u_kaby_lake.txt", FIXTURES_DIR);
    char *content = read_file_to_string(path);
    ASSERT_TRUE(content != NULL);

    arda_cpu_info_t cpu;
    memset(&cpu, 0, sizeof(cpu));
    int rc = arda_cpu_parse_proc_cpuinfo(content, strlen(content), &cpu);
    free(content);

    ASSERT_EQ_INT(rc, 0);
    ASSERT_STR_CONTAINS(cpu.vendor, "GenuineIntel");
    ASSERT_STR_CONTAINS(cpu.model_name, "i5-7200U");
    ASSERT_TRUE(cpu.has_avx);
    ASSERT_TRUE(cpu.has_avx2);
    ASSERT_TRUE(cpu.has_fma);
    ASSERT_TRUE(cpu.has_bmi1);
    ASSERT_TRUE(cpu.has_bmi2);
    ASSERT_TRUE(!cpu.has_avx512f);
    ASSERT_TRUE(!cpu.has_avx_vnni);
}

static void test_haswell_fixture(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/cpuinfo/i7_4770_haswell.txt", FIXTURES_DIR);
    char *content = read_file_to_string(path);
    ASSERT_TRUE(content != NULL);

    arda_cpu_info_t cpu;
    memset(&cpu, 0, sizeof(cpu));
    int rc = arda_cpu_parse_proc_cpuinfo(content, strlen(content), &cpu);
    free(content);

    ASSERT_EQ_INT(rc, 0);
    ASSERT_STR_CONTAINS(cpu.model_name, "i7-4770");
    ASSERT_TRUE(cpu.has_avx2);
    ASSERT_TRUE(!cpu.has_avx512f);
    ASSERT_TRUE(!cpu.has_avx_vnni);
}

static void test_zen4_fixture(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/cpuinfo/zen4_7950x.txt", FIXTURES_DIR);
    char *content = read_file_to_string(path);
    ASSERT_TRUE(content != NULL);

    arda_cpu_info_t cpu;
    memset(&cpu, 0, sizeof(cpu));
    int rc = arda_cpu_parse_proc_cpuinfo(content, strlen(content), &cpu);
    free(content);

    ASSERT_EQ_INT(rc, 0);
    ASSERT_STR_CONTAINS(cpu.vendor, "AuthenticAMD");
    ASSERT_STR_CONTAINS(cpu.model_name, "7950X");
    ASSERT_TRUE(cpu.has_avx2);
    ASSERT_TRUE(cpu.has_avx512f);
    ASSERT_TRUE(cpu.has_avx512_vnni);
    ASSERT_TRUE(cpu.has_avx512vl);
    ASSERT_TRUE(cpu.has_avx512bw);
}

static void test_alder_lake_fixture(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/cpuinfo/alder_lake_i7_12700k.txt", FIXTURES_DIR);
    char *content = read_file_to_string(path);
    ASSERT_TRUE(content != NULL);

    arda_cpu_info_t cpu;
    memset(&cpu, 0, sizeof(cpu));
    int rc = arda_cpu_parse_proc_cpuinfo(content, strlen(content), &cpu);
    free(content);

    ASSERT_EQ_INT(rc, 0);
    ASSERT_STR_CONTAINS(cpu.model_name, "12700K");
    ASSERT_TRUE(cpu.has_avx2);
    ASSERT_TRUE(cpu.has_avx_vnni);
    ASSERT_TRUE(!cpu.has_avx512f);
}

static void test_arm64_fixture(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/cpuinfo/cortex_a76_arm.txt", FIXTURES_DIR);
    char *content = read_file_to_string(path);
    ASSERT_TRUE(content != NULL);

    arda_cpu_info_t cpu;
    memset(&cpu, 0, sizeof(cpu));
    int rc = arda_cpu_parse_proc_cpuinfo(content, strlen(content), &cpu);
    free(content);

    ASSERT_EQ_INT(rc, 0);
    ASSERT_TRUE(cpu.has_neon);
    ASSERT_TRUE(cpu.has_arm_sve);
    ASSERT_TRUE(cpu.has_arm_dotprod);
}

static void test_malformed_fixture(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/cpuinfo/malformed.txt", FIXTURES_DIR);
    char *content = read_file_to_string(path);
    ASSERT_TRUE(content != NULL);

    arda_cpu_info_t cpu;
    memset(&cpu, 0, sizeof(cpu));
    int rc = arda_cpu_parse_proc_cpuinfo(content, strlen(content), &cpu);
    free(content);

    /* Must tolerate malformed without crashing */
    ASSERT_EQ_INT(rc, 0);
}

static void test_null_buffer(void) {
    arda_cpu_info_t cpu;
    memset(&cpu, 0, sizeof(cpu));
    int rc = arda_cpu_parse_proc_cpuinfo(NULL, 0, &cpu);
    ASSERT_EQ_INT(rc, -1);
}

int main(void) {
    printf("[Unit Test] CPU Detection & Parsing\n");
    RUN_TEST(test_kaby_lake_fixture);
    RUN_TEST(test_haswell_fixture);
    RUN_TEST(test_zen4_fixture);
    RUN_TEST(test_alder_lake_fixture);
    RUN_TEST(test_arm64_fixture);
    RUN_TEST(test_malformed_fixture);
    RUN_TEST(test_null_buffer);
    TEST_REPORT_MAIN();
}
