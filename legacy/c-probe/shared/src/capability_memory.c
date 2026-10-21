#include "arda/capability.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysinfo.h>
#include <unistd.h>

static void trim_whitespace(char *str) {
    if (!str) return;
    char *end;

    char *start = str;
    while (isspace((unsigned char)*start)) start++;
    if (start != str) {
        memmove(str, start, strlen(start) + 1);
    }

    end = str + strlen(str) - 1;
    while (end >= str && isspace((unsigned char)*end)) {
        *end = '\0';
        end--;
    }
}

int arda_memory_parse_proc_meminfo(const char *buffer, size_t len, arda_memory_info_t *out_mem) {
    if (!buffer || !out_mem) return -1;
    (void)len;

    char line[256];
    const char *ptr = buffer;

    uint64_t mem_total_kb = 0;
    uint64_t mem_free_kb = 0;
    uint64_t mem_available_kb = 0;
    uint64_t buffers_kb = 0;
    uint64_t cached_kb = 0;
    uint64_t swap_total_kb = 0;
    uint64_t swap_free_kb = 0;
    bool has_available = false;

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

        uint64_t kb_val = (uint64_t)strtoull(val, NULL, 10);

        if (strcmp(key, "MemTotal") == 0) {
            mem_total_kb = kb_val;
        } else if (strcmp(key, "MemFree") == 0) {
            mem_free_kb = kb_val;
        } else if (strcmp(key, "MemAvailable") == 0) {
            mem_available_kb = kb_val;
            has_available = true;
        } else if (strcmp(key, "Buffers") == 0) {
            buffers_kb = kb_val;
        } else if (strcmp(key, "Cached") == 0) {
            cached_kb = kb_val;
        } else if (strcmp(key, "SwapTotal") == 0) {
            swap_total_kb = kb_val;
        } else if (strcmp(key, "SwapFree") == 0) {
            swap_free_kb = kb_val;
        }
    }

    /* Fallback estimation if MemAvailable was missing (kernels < 3.14) */
    if (!has_available && mem_total_kb > 0) {
        mem_available_kb = mem_free_kb + buffers_kb + cached_kb;
        if (mem_available_kb > mem_total_kb) {
            mem_available_kb = mem_total_kb;
        }
    }

    out_mem->total_ram_bytes = mem_total_kb * 1024ULL;
    out_mem->available_ram_bytes = mem_available_kb * 1024ULL;
    out_mem->free_ram_bytes = mem_free_kb * 1024ULL;
    out_mem->total_swap_bytes = swap_total_kb * 1024ULL;
    out_mem->free_swap_bytes = swap_free_kb * 1024ULL;

    if (out_mem->page_size_bytes == 0) {
        long ps = sysconf(_SC_PAGESIZE);
        out_mem->page_size_bytes = (ps > 0) ? (size_t)ps : 4096;
    }

    return 0;
}

void arda_memory_probe(arda_memory_info_t *out_mem) {
    if (!out_mem) return;
    memset(out_mem, 0, sizeof(*out_mem));

    long ps = sysconf(_SC_PAGESIZE);
    out_mem->page_size_bytes = (ps > 0) ? (size_t)ps : 4096;

    /* First try reading /proc/meminfo */
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        char buffer[16384];
        size_t bytes = fread(buffer, 1, sizeof(buffer) - 1, f);
        fclose(f);
        if (bytes > 0) {
            buffer[bytes] = '\0';
            if (arda_memory_parse_proc_meminfo(buffer, bytes, out_mem) == 0 && out_mem->total_ram_bytes > 0) {
                return;
            }
        }
    }

    /* Fallback to sysinfo(2) if /proc/meminfo is inaccessible */
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        uint64_t unit = (uint64_t)si.mem_unit;
        out_mem->total_ram_bytes = (uint64_t)si.totalram * unit;
        out_mem->free_ram_bytes = (uint64_t)si.freeram * unit;
        out_mem->available_ram_bytes = out_mem->free_ram_bytes + ((uint64_t)si.bufferram * unit);
        out_mem->total_swap_bytes = (uint64_t)si.totalswap * unit;
        out_mem->free_swap_bytes = (uint64_t)si.freeswap * unit;
    }
}
