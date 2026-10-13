#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_tests_run = 0;
static int g_tests_failed = 0;

#define ASSERT_TRUE(cond) do { \
    g_tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s:%d: assertion failed: %s\n", __FILE__, __LINE__, #cond); \
        g_tests_failed++; \
        return; \
    } \
} while (0)

#define ASSERT_EQ_INT(a, b) do { \
    g_tests_run++; \
    if ((a) != (b)) { \
        fprintf(stderr, "FAIL: %s:%d: %s (%d) != %s (%d)\n", __FILE__, __LINE__, #a, (int)(a), #b, (int)(b)); \
        g_tests_failed++; \
        return; \
    } \
} while (0)

#define ASSERT_EQ_UINT64(a, b) do { \
    g_tests_run++; \
    if ((a) != (b)) { \
        fprintf(stderr, "FAIL: %s:%d: %s (%llu) != %s (%llu)\n", __FILE__, __LINE__, #a, (unsigned long long)(a), #b, (unsigned long long)(b)); \
        g_tests_failed++; \
        return; \
    } \
} while (0)

#define ASSERT_STR_CONTAINS(haystack, needle) do { \
    g_tests_run++; \
    if (!strstr((haystack), (needle))) { \
        fprintf(stderr, "FAIL: %s:%d: string '%s' does not contain '%s'\n", __FILE__, __LINE__, (haystack), (needle)); \
        g_tests_failed++; \
        return; \
    } \
} while (0)

#define RUN_TEST(fn) do { \
    printf("  Running %s...", #fn); \
    int before_fail = g_tests_failed; \
    fn(); \
    if (g_tests_failed == before_fail) { \
        printf(" OK\n"); \
    } \
} while (0)

#define TEST_REPORT_MAIN() do { \
    printf("\nTest summary: %d checks run, %d failures\n", g_tests_run, g_tests_failed); \
    return (g_tests_failed == 0) ? 0 : 1; \
} while (0)

static inline __attribute__((unused)) char *read_file_to_string(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    size_t length = (size_t)sz;
    char *buf = (char *)malloc(length + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t read_bytes = fread(buf, 1, length, f);
    buf[read_bytes] = '\0';
    fclose(f);
    return buf;
}

#endif /* TEST_HELPERS_H */
