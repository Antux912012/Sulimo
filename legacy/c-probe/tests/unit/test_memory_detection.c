#include "arda/capability.h"
#include "../test_helpers.h"

#ifndef FIXTURES_DIR
#define FIXTURES_DIR "fixtures"
#endif

static void test_normal_16gb_fixture(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/meminfo/normal_16gb.txt", FIXTURES_DIR);
    char *content = read_file_to_string(path);
    ASSERT_TRUE(content != NULL);

    arda_memory_info_t mem;
    memset(&mem, 0, sizeof(mem));
    int rc = arda_memory_parse_proc_meminfo(content, strlen(content), &mem);
    free(content);

    ASSERT_EQ_INT(rc, 0);
    ASSERT_TRUE(mem.total_ram_bytes > 0);
    ASSERT_TRUE(mem.available_ram_bytes > 0);
    ASSERT_TRUE(mem.free_ram_bytes > 0);
    ASSERT_TRUE(mem.page_size_bytes >= 4096);
}

static void test_constrained_8gb_fixture(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/meminfo/constrained_8gb.txt", FIXTURES_DIR);
    char *content = read_file_to_string(path);
    ASSERT_TRUE(content != NULL);

    arda_memory_info_t mem;
    memset(&mem, 0, sizeof(mem));
    int rc = arda_memory_parse_proc_meminfo(content, strlen(content), &mem);
    free(content);

    ASSERT_EQ_INT(rc, 0);
    ASSERT_EQ_UINT64(mem.total_ram_bytes, 8192000ULL * 1024ULL);
    ASSERT_EQ_UINT64(mem.available_ram_bytes, 4096000ULL * 1024ULL);
    ASSERT_EQ_UINT64(mem.free_ram_bytes, 2048000ULL * 1024ULL);
    ASSERT_EQ_UINT64(mem.total_swap_bytes, 4096000ULL * 1024ULL);
}

static void test_legacy_no_memavailable_fixture(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/meminfo/legacy_no_memavailable.txt", FIXTURES_DIR);
    char *content = read_file_to_string(path);
    ASSERT_TRUE(content != NULL);

    arda_memory_info_t mem;
    memset(&mem, 0, sizeof(mem));
    int rc = arda_memory_parse_proc_meminfo(content, strlen(content), &mem);
    free(content);

    ASSERT_EQ_INT(rc, 0);
    ASSERT_EQ_UINT64(mem.total_ram_bytes, 16384000ULL * 1024ULL);
    /* Available estimated as Free + Buffers + Cached = 4096000 + 512000 + 6144000 = 10752000 kB */
    uint64_t expected_avail = (4096000ULL + 512000ULL + 6144000ULL) * 1024ULL;
    ASSERT_EQ_UINT64(mem.available_ram_bytes, expected_avail);
}

static void test_malformed_meminfo_fixture(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/meminfo/malformed.txt", FIXTURES_DIR);
    char *content = read_file_to_string(path);
    ASSERT_TRUE(content != NULL);

    arda_memory_info_t mem;
    memset(&mem, 0, sizeof(mem));
    int rc = arda_memory_parse_proc_meminfo(content, strlen(content), &mem);
    free(content);

    /* Must tolerate malformed text safely without crashing */
    ASSERT_EQ_INT(rc, 0);
}

static void test_null_buffer(void) {
    arda_memory_info_t mem;
    memset(&mem, 0, sizeof(mem));
    int rc = arda_memory_parse_proc_meminfo(NULL, 0, &mem);
    ASSERT_EQ_INT(rc, -1);
}

int main(void) {
    printf("[Unit Test] Memory Detection & Parsing\n");
    RUN_TEST(test_normal_16gb_fixture);
    RUN_TEST(test_constrained_8gb_fixture);
    RUN_TEST(test_legacy_no_memavailable_fixture);
    RUN_TEST(test_malformed_meminfo_fixture);
    RUN_TEST(test_null_buffer);
    TEST_REPORT_MAIN();
}
