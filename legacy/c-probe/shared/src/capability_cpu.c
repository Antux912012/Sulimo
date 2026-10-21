#include "arda/capability.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <cpuid.h>
#define ARDA_ARCH_X86 1
#else
#define ARDA_ARCH_X86 0
#endif

/* Checks if a space-separated flag string contains a specific token */
static bool flag_token_present(const char *flags, const char *token) {
    if (!flags || !token) return false;
    size_t tok_len = strlen(token);
    const char *p = flags;

    while (*p) {
        /* Skip leading whitespace */
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;

        const char *end = p;
        while (*end && !isspace((unsigned char)*end)) end++;

        if ((size_t)(end - p) == tok_len && strncmp(p, token, tok_len) == 0) {
            return true;
        }
        p = end;
    }
    return false;
}

static void trim_whitespace(char *str) {
    if (!str) return;
    char *end;

    /* Trim leading space */
    char *start = str;
    while (isspace((unsigned char)*start)) start++;

    if (start != str) {
        memmove(str, start, strlen(start) + 1);
    }

    /* Trim trailing space */
    end = str + strlen(str) - 1;
    while (end >= str && isspace((unsigned char)*end)) {
        *end = '\0';
        end--;
    }
}

int arda_cpu_parse_proc_cpuinfo(const char *buffer, size_t len, arda_cpu_info_t *out_cpu) {
    if (!buffer || !out_cpu) return -1;
    (void)len;

    char line[8192];
    const char *ptr = buffer;
    uint32_t processor_count = 0;
    bool found_flags = false;

    while (*ptr) {
        size_t line_len = 0;
        while (*ptr && *ptr != '\n') {
            if (line_len < sizeof(line) - 1) {
                line[line_len++] = *ptr;
            }
            ptr++;
        }
        line[line_len] = '\0';
        if (*ptr == '\n') ptr++;

        char *colon = strchr(line, ':');
        if (!colon) continue;

        *colon = '\0';
        char *key = line;
        char *val = colon + 1;
        trim_whitespace(key);
        trim_whitespace(val);

        if (strcmp(key, "processor") == 0) {
            processor_count++;
        } else if (strcmp(key, "vendor_id") == 0 || strcmp(key, "CPU implementer") == 0) {
            if (out_cpu->vendor[0] == '\0') {
                strncpy(out_cpu->vendor, val, sizeof(out_cpu->vendor) - 1);
            }
        } else if (strcmp(key, "model name") == 0 || strcmp(key, "Hardware") == 0 || strcmp(key, "Model") == 0) {
            if (out_cpu->model_name[0] == '\0') {
                strncpy(out_cpu->model_name, val, sizeof(out_cpu->model_name) - 1);
            }
        } else if (strcmp(key, "cpu cores") == 0) {
            if (out_cpu->physical_cores == 0) {
                out_cpu->physical_cores = (uint32_t)atoi(val);
            }
        } else if (strcmp(key, "flags") == 0 || strcmp(key, "Features") == 0) {
            /* If multiple cores are listed, parse flags from the first core only */
            if (!found_flags) {
                found_flags = true;
                out_cpu->has_sse3 = flag_token_present(val, "pni") || flag_token_present(val, "sse3");
                out_cpu->has_ssse3 = flag_token_present(val, "ssse3");
                out_cpu->has_sse4_1 = flag_token_present(val, "sse4_1");
                out_cpu->has_sse4_2 = flag_token_present(val, "sse4_2");
                out_cpu->has_avx = flag_token_present(val, "avx");
                out_cpu->has_f16c = flag_token_present(val, "f16c");
                out_cpu->has_fma = flag_token_present(val, "fma");
                out_cpu->has_avx2 = flag_token_present(val, "avx2");
                out_cpu->has_bmi1 = flag_token_present(val, "bmi1");
                out_cpu->has_bmi2 = flag_token_present(val, "bmi2");
                out_cpu->has_avx512f = flag_token_present(val, "avx512f");
                out_cpu->has_avx512cd = flag_token_present(val, "avx512cd");
                out_cpu->has_avx512bw = flag_token_present(val, "avx512bw");
                out_cpu->has_avx512dq = flag_token_present(val, "avx512dq");
                out_cpu->has_avx512vl = flag_token_present(val, "avx512vl");
                out_cpu->has_avx512_vnni = flag_token_present(val, "avx512_vnni");
                out_cpu->has_avx_vnni = flag_token_present(val, "avx_vnni");
                out_cpu->has_amx_tile = flag_token_present(val, "amx_tile");
                out_cpu->has_amx_int8 = flag_token_present(val, "amx_int8");
                out_cpu->has_amx_bf16 = flag_token_present(val, "amx_bf16");

                /* ARM features */
                out_cpu->has_neon = flag_token_present(val, "neon") || flag_token_present(val, "asimd");
                out_cpu->has_arm_fp16 = flag_token_present(val, "fp16") || flag_token_present(val, "half");
                out_cpu->has_arm_dotprod = flag_token_present(val, "asimddp") || flag_token_present(val, "dotprod");
                out_cpu->has_arm_sve = flag_token_present(val, "sve");
            }
        }
    }

    if (processor_count > 0 && out_cpu->logical_cores == 0) {
        out_cpu->logical_cores = processor_count;
    }

    return 0;
}

#if ARDA_ARCH_X86
static void probe_x86_cpuid(arda_cpu_info_t *cpu) {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

    if (!__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
        return;
    }

    unsigned int max_level = eax;

    /* Vendor ID */
    char vendor[13];
    memcpy(vendor + 0, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';
    if (cpu->vendor[0] == '\0') {
        strncpy(cpu->vendor, vendor, sizeof(cpu->vendor) - 1);
    }

    /* Model Name via extended CPUID */
    if (__get_cpuid(0x80000000, &eax, &ebx, &ecx, &edx)) {
        unsigned int max_ext = eax;
        if (max_ext >= 0x80000004) {
            char brand[49];
            memset(brand, 0, sizeof(brand));
            __get_cpuid(0x80000002, (unsigned int *)(brand + 0),  (unsigned int *)(brand + 4),
                                    (unsigned int *)(brand + 8),  (unsigned int *)(brand + 12));
            __get_cpuid(0x80000003, (unsigned int *)(brand + 16), (unsigned int *)(brand + 20),
                                    (unsigned int *)(brand + 24), (unsigned int *)(brand + 28));
            __get_cpuid(0x80000004, (unsigned int *)(brand + 32), (unsigned int *)(brand + 36),
                                    (unsigned int *)(brand + 40), (unsigned int *)(brand + 44));
            trim_whitespace(brand);
            if (cpu->model_name[0] == '\0' && brand[0] != '\0') {
                strncpy(cpu->model_name, brand, sizeof(cpu->model_name) - 1);
            }
        }
    }

    /* Basic features: Level 1 */
    if (max_level >= 1) {
        __get_cpuid(1, &eax, &ebx, &ecx, &edx);
        cpu->has_sse3   = (ecx & (1u << 0)) != 0;
        cpu->has_ssse3  = (ecx & (1u << 9)) != 0;
        cpu->has_fma    = (ecx & (1u << 12)) != 0;
        cpu->has_sse4_1 = (ecx & (1u << 19)) != 0;
        cpu->has_sse4_2 = (ecx & (1u << 20)) != 0;
        cpu->has_avx    = (ecx & (1u << 28)) != 0;
        cpu->has_f16c   = (ecx & (1u << 29)) != 0;
    }

    /* Structured extended features: Level 7, sublevel 0 */
    if (max_level >= 7) {
        __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
        cpu->has_bmi1       = (ebx & (1u << 3)) != 0;
        cpu->has_avx2       = (ebx & (1u << 5)) != 0;
        cpu->has_bmi2       = (ebx & (1u << 8)) != 0;
        cpu->has_avx512f    = (ebx & (1u << 16)) != 0;
        cpu->has_avx512dq   = (ebx & (1u << 17)) != 0;
        cpu->has_avx512cd   = (ebx & (1u << 28)) != 0;
        cpu->has_avx512bw   = (ebx & (1u << 30)) != 0;
        cpu->has_avx512vl   = (ebx & (1u << 31)) != 0;

        cpu->has_avx512_vnni = (ecx & (1u << 11)) != 0;

        cpu->has_amx_bf16   = (edx & (1u << 22)) != 0;
        cpu->has_amx_tile   = (edx & (1u << 24)) != 0;
        cpu->has_amx_int8   = (edx & (1u << 25)) != 0;

        /* Level 7, sublevel 1 (AVX-VNNI) */
        __get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx);
        cpu->has_avx_vnni   = (eax & (1u << 4)) != 0;
    }
}
#endif

void arda_cpu_probe(arda_cpu_info_t *out_cpu) {
    if (!out_cpu) return;
    memset(out_cpu, 0, sizeof(*out_cpu));

    /* Logical processors from system configuration */
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nprocs > 0) {
        out_cpu->logical_cores = (uint32_t)nprocs;
    }

#if ARDA_ARCH_X86
    /* Hardware CPUID inquiry */
    probe_x86_cpuid(out_cpu);
#endif

    /* Supplement / fallback with /proc/cpuinfo */
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char buffer[32768];
        size_t bytes = fread(buffer, 1, sizeof(buffer) - 1, f);
        fclose(f);
        if (bytes > 0) {
            buffer[bytes] = '\0';
            arda_cpu_parse_proc_cpuinfo(buffer, bytes, out_cpu);
        }
    }

    if (out_cpu->vendor[0] == '\0') {
        strncpy(out_cpu->vendor, "Unknown", sizeof(out_cpu->vendor) - 1);
    }
    if (out_cpu->model_name[0] == '\0') {
        strncpy(out_cpu->model_name, "Generic CPU", sizeof(out_cpu->model_name) - 1);
    }
    if (out_cpu->logical_cores == 0) {
        out_cpu->logical_cores = 1;
    }
    if (out_cpu->physical_cores == 0) {
        out_cpu->physical_cores = out_cpu->logical_cores;
    }
}
