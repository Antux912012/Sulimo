#include "arda/capability.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int arda_capability_report_collect(arda_capability_report_t *out_report) {
    if (!out_report) return -1;
    memset(out_report, 0, sizeof(*out_report));

    arda_os_probe(&out_report->os);
    arda_cpu_probe(&out_report->cpu);
    arda_memory_probe(&out_report->memory);
    arda_backends_probe(out_report->backends, ARDA_MAX_BACKENDS, &out_report->backend_count);
    arda_profile_evaluate(&out_report->cpu, &out_report->memory, out_report->backends,
                          out_report->backend_count, &out_report->selected_profile);

    return 0;
}

int arda_capability_report_collect_from_proc(
    arda_capability_report_t *out_report,
    const char *cpuinfo_path,
    const char *meminfo_path
) {
    if (!out_report) return -1;
    memset(out_report, 0, sizeof(*out_report));

    arda_os_probe(&out_report->os);

    if (cpuinfo_path) {
        FILE *f = fopen(cpuinfo_path, "r");
        if (f) {
            char buf[65536];
            size_t n = fread(buf, 1, sizeof(buf) - 1, f);
            fclose(f);
            if (n > 0) {
                buf[n] = '\0';
                arda_cpu_parse_proc_cpuinfo(buf, n, &out_report->cpu);
            }
        }
    } else {
        arda_cpu_probe(&out_report->cpu);
    }

    if (meminfo_path) {
        FILE *f = fopen(meminfo_path, "r");
        if (f) {
            char buf[16384];
            size_t n = fread(buf, 1, sizeof(buf) - 1, f);
            fclose(f);
            if (n > 0) {
                buf[n] = '\0';
                arda_memory_parse_proc_meminfo(buf, n, &out_report->memory);
            }
        }
    } else {
        arda_memory_probe(&out_report->memory);
    }

    arda_backends_probe(out_report->backends, ARDA_MAX_BACKENDS, &out_report->backend_count);
    arda_profile_evaluate(&out_report->cpu, &out_report->memory, out_report->backends,
                          out_report->backend_count, &out_report->selected_profile);

    return 0;
}

int arda_capability_report_format_text(const arda_capability_report_t *report, char *buf, size_t buf_size) {
    if (!report || !buf || buf_size == 0) return -1;

    size_t offset = 0;
    int written = 0;

#define APPEND(...) do { \
    written = snprintf(buf + offset, buf_size - offset, __VA_ARGS__); \
    if (written < 0 || (size_t)written >= buf_size - offset) return -1; \
    offset += (size_t)written; \
} while(0)

    APPEND("======================================================================\n");
    APPEND("                   ARDA'S BREATH - CAPABILITY REPORT                  \n");
    APPEND("======================================================================\n\n");

    /* OS & Host */
    APPEND("[Operating System]\n");
    APPEND("  OS:           %s %s\n", report->os.sysname, report->os.release);
    APPEND("  Architecture: %s\n", report->os.machine);
    APPEND("  Kernel Info:  %s\n\n", report->os.version);

    /* CPU */
    APPEND("[CPU Architecture & Features]\n");
    APPEND("  Vendor:       %s\n", report->cpu.vendor);
    APPEND("  Model:        %s\n", report->cpu.model_name);
    APPEND("  Cores:        %u logical, %u physical\n", report->cpu.logical_cores, report->cpu.physical_cores);
    APPEND("  Baseline SIMD: SSE3=%s, SSSE3=%s, SSE4.1=%s, SSE4.2=%s\n",
           report->cpu.has_sse3 ? "yes" : "no",
           report->cpu.has_ssse3 ? "yes" : "no",
           report->cpu.has_sse4_1 ? "yes" : "no",
           report->cpu.has_sse4_2 ? "yes" : "no");
    APPEND("  AVX / FMA:    AVX=%s, F16C=%s, FMA=%s, AVX2=%s, BMI1=%s, BMI2=%s\n",
           report->cpu.has_avx ? "yes" : "no",
           report->cpu.has_f16c ? "yes" : "no",
           report->cpu.has_fma ? "yes" : "no",
           report->cpu.has_avx2 ? "yes" : "no",
           report->cpu.has_bmi1 ? "yes" : "no",
           report->cpu.has_bmi2 ? "yes" : "no");
    APPEND("  AVX-512:      F=%s, CD=%s, BW=%s, DQ=%s, VL=%s, VNNI=%s\n",
           report->cpu.has_avx512f ? "yes" : "no",
           report->cpu.has_avx512cd ? "yes" : "no",
           report->cpu.has_avx512bw ? "yes" : "no",
           report->cpu.has_avx512dq ? "yes" : "no",
           report->cpu.has_avx512vl ? "yes" : "no",
           report->cpu.has_avx512_vnni ? "yes" : "no");
    APPEND("  Advanced AI:  AVX-VNNI=%s, AMX-TILE=%s, AMX-INT8=%s, AMX-BF16=%s\n",
           report->cpu.has_avx_vnni ? "yes" : "no",
           report->cpu.has_amx_tile ? "yes" : "no",
           report->cpu.has_amx_int8 ? "yes" : "no",
           report->cpu.has_amx_bf16 ? "yes" : "no");
    if (report->cpu.has_neon || report->cpu.has_arm_sve) {
        APPEND("  ARM Features: NEON=%s, FP16=%s, DotProd=%s, SVE=%s\n",
               report->cpu.has_neon ? "yes" : "no",
               report->cpu.has_arm_fp16 ? "yes" : "no",
               report->cpu.has_arm_dotprod ? "yes" : "no",
               report->cpu.has_arm_sve ? "yes" : "no");
    }
    APPEND("\n");

    /* Memory */
    double total_ram_mb = (double)report->memory.total_ram_bytes / (1024.0 * 1024.0);
    double avail_ram_mb = (double)report->memory.available_ram_bytes / (1024.0 * 1024.0);
    double free_ram_mb = (double)report->memory.free_ram_bytes / (1024.0 * 1024.0);
    double swap_total_mb = (double)report->memory.total_swap_bytes / (1024.0 * 1024.0);
    double swap_free_mb = (double)report->memory.free_swap_bytes / (1024.0 * 1024.0);

    APPEND("[System Memory]\n");
    APPEND("  Total RAM:    %.1f MB (%.2f GB)\n", total_ram_mb, total_ram_mb / 1024.0);
    APPEND("  Available:    %.1f MB (%.2f GB)\n", avail_ram_mb, avail_ram_mb / 1024.0);
    APPEND("  Free RAM:     %.1f MB (%.2f GB)\n", free_ram_mb, free_ram_mb / 1024.0);
    APPEND("  Swap:         %.1f MB total, %.1f MB free\n", swap_total_mb, swap_free_mb);
    APPEND("  Page Size:    %zu bytes\n\n", report->memory.page_size_bytes);

    /* Backends */
    APPEND("[Inference & Acceleration Backends]\n");
    for (size_t i = 0; i < report->backend_count; i++) {
        const arda_backend_status_t *b = &report->backends[i];
        APPEND("  Backend: %s\n", b->name);
        APPEND("    Compiled in:       %s\n", b->compiled_in ? "yes" : "no");
        APPEND("    Runtime available: %s\n", b->runtime_available ? "yes" : "no");
        if (b->error_reason[0] != '\0') {
            APPEND("    Diagnostic/Status: %s\n", b->error_reason);
        }
        if (b->device_count > 0) {
            APPEND("    Devices (%u):\n", b->device_count);
            for (uint32_t d = 0; d < b->device_count; d++) {
                const arda_gpu_device_t *gpu = &b->devices[d];
                double vram_mb = (double)gpu->total_memory_bytes / (1024.0 * 1024.0);
                const char *dev_type = gpu->is_discrete ? "Discrete" :
                                       (gpu->is_integrated ? "Integrated" :
                                       (gpu->is_software ? "Software Emulated" : "System"));
                const char *vendor_str = gpu->vendor[0] ? gpu->vendor : "Host";
                if (vram_mb > 0.0) {
                    APPEND("      [%u] %s (Vendor: %s, %s, VRAM: %.0f MB)\n",
                           d, gpu->name, vendor_str, dev_type, vram_mb);
                } else {
                    APPEND("      [%u] %s (Vendor: %s, %s)\n",
                           d, gpu->name, vendor_str, dev_type);
                }
            }
        }
    }
    APPEND("\n");

    /* Selected Profile */
    APPEND("[Runtime Policy Profile]\n");
    APPEND("  Selected Profile: %s\n", report->selected_profile.name);
    APPEND("  Selection Reason: %s\n", report->selected_profile.reason);
    APPEND("======================================================================\n");

#undef APPEND

    return 0;
}

int arda_capability_report_format_json(const arda_capability_report_t *report, char *buf, size_t buf_size) {
    if (!report || !buf || buf_size == 0) return -1;

    size_t offset = 0;
    int written = 0;

#define APPEND(...) do { \
    written = snprintf(buf + offset, buf_size - offset, __VA_ARGS__); \
    if (written < 0 || (size_t)written >= buf_size - offset) return -1; \
    offset += (size_t)written; \
} while(0)

    APPEND("{\n");
    APPEND("  \"os\": {\n");
    APPEND("    \"sysname\": \"%s\",\n", report->os.sysname);
    APPEND("    \"release\": \"%s\",\n", report->os.release);
    APPEND("    \"machine\": \"%s\"\n", report->os.machine);
    APPEND("  },\n");

    APPEND("  \"cpu\": {\n");
    APPEND("    \"vendor\": \"%s\",\n", report->cpu.vendor);
    APPEND("    \"model_name\": \"%s\",\n", report->cpu.model_name);
    APPEND("    \"logical_cores\": %u,\n", report->cpu.logical_cores);
    APPEND("    \"physical_cores\": %u,\n", report->cpu.physical_cores);
    APPEND("    \"features\": {\n");
    APPEND("      \"sse3\": %s,\n", report->cpu.has_sse3 ? "true" : "false");
    APPEND("      \"ssse3\": %s,\n", report->cpu.has_ssse3 ? "true" : "false");
    APPEND("      \"sse4_1\": %s,\n", report->cpu.has_sse4_1 ? "true" : "false");
    APPEND("      \"sse4_2\": %s,\n", report->cpu.has_sse4_2 ? "true" : "false");
    APPEND("      \"avx\": %s,\n", report->cpu.has_avx ? "true" : "false");
    APPEND("      \"f16c\": %s,\n", report->cpu.has_f16c ? "true" : "false");
    APPEND("      \"fma\": %s,\n", report->cpu.has_fma ? "true" : "false");
    APPEND("      \"avx2\": %s,\n", report->cpu.has_avx2 ? "true" : "false");
    APPEND("      \"bmi1\": %s,\n", report->cpu.has_bmi1 ? "true" : "false");
    APPEND("      \"bmi2\": %s,\n", report->cpu.has_bmi2 ? "true" : "false");
    APPEND("      \"avx512f\": %s,\n", report->cpu.has_avx512f ? "true" : "false");
    APPEND("      \"avx512_vnni\": %s,\n", report->cpu.has_avx512_vnni ? "true" : "false");
    APPEND("      \"avx_vnni\": %s,\n", report->cpu.has_avx_vnni ? "true" : "false");
    APPEND("      \"amx_tile\": %s\n", report->cpu.has_amx_tile ? "true" : "false");
    APPEND("    }\n");
    APPEND("  },\n");

    APPEND("  \"memory\": {\n");
    APPEND("    \"total_ram_bytes\": %llu,\n", (unsigned long long)report->memory.total_ram_bytes);
    APPEND("    \"available_ram_bytes\": %llu,\n", (unsigned long long)report->memory.available_ram_bytes);
    APPEND("    \"free_ram_bytes\": %llu,\n", (unsigned long long)report->memory.free_ram_bytes);
    APPEND("    \"total_swap_bytes\": %llu,\n", (unsigned long long)report->memory.total_swap_bytes);
    APPEND("    \"free_swap_bytes\": %llu,\n", (unsigned long long)report->memory.free_swap_bytes);
    APPEND("    \"page_size_bytes\": %zu\n", report->memory.page_size_bytes);
    APPEND("  },\n");

    APPEND("  \"backends\": [\n");
    for (size_t i = 0; i < report->backend_count; i++) {
        const arda_backend_status_t *b = &report->backends[i];
        APPEND("    {\n");
        APPEND("      \"id\": %d,\n", (int)b->id);
        APPEND("      \"name\": \"%s\",\n", b->name);
        APPEND("      \"compiled_in\": %s,\n", b->compiled_in ? "true" : "false");
        APPEND("      \"runtime_available\": %s,\n", b->runtime_available ? "true" : "false");
        APPEND("      \"error_reason\": \"%s\",\n", b->error_reason);
        APPEND("      \"devices\": [\n");
        for (uint32_t d = 0; d < b->device_count; d++) {
            const arda_gpu_device_t *gpu = &b->devices[d];
            APPEND("        {\n");
            APPEND("          \"name\": \"%s\",\n", gpu->name);
            APPEND("          \"vendor\": \"%s\",\n", gpu->vendor);
            APPEND("          \"is_integrated\": %s,\n", gpu->is_integrated ? "true" : "false");
            APPEND("          \"is_discrete\": %s,\n", gpu->is_discrete ? "true" : "false");
            APPEND("          \"is_software\": %s,\n", gpu->is_software ? "true" : "false");
            APPEND("          \"total_memory_bytes\": %llu\n", (unsigned long long)gpu->total_memory_bytes);
            APPEND("        }%s\n", (d + 1 < b->device_count) ? "," : "");
        }
        APPEND("      ]\n");
        APPEND("    }%s\n", (i + 1 < report->backend_count) ? "," : "");
    }
    APPEND("  ],\n");

    APPEND("  \"profile\": {\n");
    APPEND("    \"type\": %d,\n", (int)report->selected_profile.type);
    APPEND("    \"name\": \"%s\",\n", report->selected_profile.name);
    APPEND("    \"reason\": \"%s\"\n", report->selected_profile.reason);
    APPEND("  }\n");
    APPEND("}\n");

#undef APPEND

    return 0;
}
