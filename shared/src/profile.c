#include "arda/profile.h"
#include "arda/capability.h"

#include <stdio.h>
#include <string.h>

const char *arda_profile_type_to_string(arda_profile_type_t type) {
    switch (type) {
        case ARDA_PROFILE_PORTABLE: return "arda-portable";
        case ARDA_PROFILE_CLASSIC:  return "arda-classic";
        case ARDA_PROFILE_CURRENT:  return "arda-current";
        case ARDA_PROFILE_UNKNOWN:
        default:                    return "arda-unknown";
    }
}

void arda_profile_evaluate(
    const arda_cpu_info_t *cpu,
    const arda_memory_info_t *mem,
    const arda_backend_status_t *backends,
    size_t backend_count,
    arda_profile_info_t *out_profile
) {
    if (!out_profile) return;
    memset(out_profile, 0, sizeof(*out_profile));

    if (!cpu || !mem) {
        out_profile->type = ARDA_PROFILE_PORTABLE;
        out_profile->name = arda_profile_type_to_string(out_profile->type);
        strncpy(out_profile->reason, "Missing CPU or memory capability information; defaulting to portable baseline.",
                sizeof(out_profile->reason) - 1);
        return;
    }

    double total_ram_gb = (double)mem->total_ram_bytes / (1024.0 * 1024.0 * 1024.0);

    /* Inspect GPU availability */
    bool vulkan_available = false;
    bool has_discrete_gpu = false;

    for (size_t i = 0; i < backend_count; i++) {
        if (backends[i].id == ARDA_BACKEND_VULKAN && backends[i].runtime_available) {
            vulkan_available = true;
            for (uint32_t d = 0; d < backends[i].device_count; d++) {
                if (backends[i].devices[d].is_discrete) {
                    has_discrete_gpu = true;
                }
            }
        }
    }

    /* Condition 1: Older or minimal CPU lacking AVX2 */
    if (!cpu->has_avx2 && !cpu->has_neon) {
        out_profile->type = ARDA_PROFILE_PORTABLE;
        out_profile->name = arda_profile_type_to_string(out_profile->type);
        snprintf(out_profile->reason, sizeof(out_profile->reason),
                 "CPU lacks AVX2 vector instructions. Baseline portable profile selected for compatibility.");
        return;
    }

    /* Check for modern AI-relevant vector instructions */
    bool has_advanced_simd = (cpu->has_avx512f && cpu->has_avx512_vnni) || cpu->has_avx_vnni || cpu->has_amx_tile;

    /* Condition 2: Modern CPU with AVX-512/AVX-VNNI or capable discrete GPU, and sufficient RAM */
    if ((has_advanced_simd || (vulkan_available && has_discrete_gpu)) && total_ram_gb >= 15.0) {
        out_profile->type = ARDA_PROFILE_CURRENT;
        out_profile->name = arda_profile_type_to_string(out_profile->type);

        const char *simd_desc = cpu->has_avx_vnni ? "AVX-VNNI" :
                                (cpu->has_avx512_vnni ? "AVX-512 VNNI" :
                                (cpu->has_avx512f ? "AVX-512" : "AVX2 + discrete GPU"));

        snprintf(out_profile->reason, sizeof(out_profile->reason),
                 "Modern compute capabilities detected (%s, %.1f GB RAM%s). Current profile selected for high throughput and GPU offload.",
                 simd_desc, total_ram_gb, has_discrete_gpu ? ", discrete GPU present" : "");
        return;
    }

    /* Condition 3: AVX2 workhorse systems (8-16 GB RAM, integrated GPU or SATA storage) */
    out_profile->type = ARDA_PROFILE_CLASSIC;
    out_profile->name = arda_profile_type_to_string(out_profile->type);

    snprintf(out_profile->reason, sizeof(out_profile->reason),
             "AVX2 detected without AVX-512/AVX-VNNI (RAM: %.1f GB%s%s). Classic profile selected for stable AVX2 CPU execution with conservative memory budgets.",
             total_ram_gb,
             vulkan_available ? ", Vulkan iGPU available" : ", no GPU offload",
             has_discrete_gpu ? ", discrete GPU present" : "");
}
