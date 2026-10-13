#ifndef ARDA_CAPABILITY_H
#define ARDA_CAPABILITY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "arda/profile.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ARDA_MAX_BACKENDS 8
#define ARDA_MAX_GPU_DEVICES 8

typedef enum arda_backend_id {
    ARDA_BACKEND_CPU = 0,
    ARDA_BACKEND_VULKAN = 1,
    ARDA_BACKEND_OPENCL = 2,
    ARDA_BACKEND_SYCL = 3
} arda_backend_id_t;

typedef struct arda_os_info {
    char sysname[128];
    char release[128];
    char version[256];
    char machine[128];
} arda_os_info_t;

typedef struct arda_cpu_info {
    char vendor[64];
    char model_name[128];
    uint32_t logical_cores;
    uint32_t physical_cores;

    /* x86 vector extensions */
    bool has_sse3;
    bool has_ssse3;
    bool has_sse4_1;
    bool has_sse4_2;
    bool has_avx;
    bool has_f16c;
    bool has_fma;
    bool has_avx2;
    bool has_bmi1;
    bool has_bmi2;
    bool has_avx512f;
    bool has_avx512cd;
    bool has_avx512bw;
    bool has_avx512dq;
    bool has_avx512vl;
    bool has_avx512_vnni;
    bool has_avx_vnni;
    bool has_amx_tile;
    bool has_amx_int8;
    bool has_amx_bf16;

    /* ARM extensions */
    bool has_neon;
    bool has_arm_fp16;
    bool has_arm_dotprod;
    bool has_arm_sve;
} arda_cpu_info_t;

typedef struct arda_memory_info {
    uint64_t total_ram_bytes;
    uint64_t available_ram_bytes;
    uint64_t free_ram_bytes;
    uint64_t total_swap_bytes;
    uint64_t free_swap_bytes;
    size_t page_size_bytes;
} arda_memory_info_t;

typedef struct arda_gpu_device {
    char name[256];
    char vendor[64];
    char driver_version[64];
    uint64_t total_memory_bytes;
    uint64_t free_memory_bytes;
    bool is_integrated;
    bool is_discrete;
    bool is_software;
    char pci_bus_id[32];
} arda_gpu_device_t;

typedef struct arda_backend_status {
    arda_backend_id_t id;
    const char *name;
    bool compiled_in;
    bool runtime_available;
    char error_reason[256];
    uint32_t device_count;
    arda_gpu_device_t devices[ARDA_MAX_GPU_DEVICES];
} arda_backend_status_t;

typedef struct arda_capability_report {
    arda_os_info_t os;
    arda_cpu_info_t cpu;
    arda_memory_info_t memory;
    arda_backend_status_t backends[ARDA_MAX_BACKENDS];
    size_t backend_count;
    arda_profile_info_t selected_profile;
} arda_capability_report_t;

/* Subsystem probes */
void arda_os_probe(arda_os_info_t *out_os);
void arda_cpu_probe(arda_cpu_info_t *out_cpu);
int arda_cpu_parse_proc_cpuinfo(const char *buffer, size_t len, arda_cpu_info_t *out_cpu);

void arda_memory_probe(arda_memory_info_t *out_mem);
int arda_memory_parse_proc_meminfo(const char *buffer, size_t len, arda_memory_info_t *out_mem);

void arda_backends_probe(arda_backend_status_t *out_backends, size_t max_backends, size_t *out_count);

/* Full report collection */
int arda_capability_report_collect(arda_capability_report_t *out_report);
int arda_capability_report_collect_from_proc(
    arda_capability_report_t *out_report,
    const char *cpuinfo_path,
    const char *meminfo_path
);

/* Report formatting */
int arda_capability_report_format_text(const arda_capability_report_t *report, char *buf, size_t buf_size);
int arda_capability_report_format_json(const arda_capability_report_t *report, char *buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif /* ARDA_CAPABILITY_H */
