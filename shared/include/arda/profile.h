#ifndef ARDA_PROFILE_H
#define ARDA_PROFILE_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum arda_profile_type {
    ARDA_PROFILE_UNKNOWN = 0,
    ARDA_PROFILE_PORTABLE = 1, /* Baseline/fallback: SSE4/portable CPU, conservative RAM */
    ARDA_PROFILE_CLASSIC = 2,  /* AVX2 workhorse, 8-16 GB RAM, conservative offload */
    ARDA_PROFILE_CURRENT = 3   /* AVX-512 / AVX-VNNI, 16-32+ GB RAM, full offload */
} arda_profile_type_t;

typedef struct arda_profile_info {
    arda_profile_type_t type;
    const char *name;
    char reason[512];
} arda_profile_info_t;

/* Forward declarations of capability structures */
struct arda_cpu_info;
struct arda_memory_info;
struct arda_backend_status;

/**
 * Evaluates the runtime hardware capabilities and selects an appropriate policy profile.
 *
 * Rules:
 * 1. Does NOT assume capability based on CPU model/generation name strings.
 * 2. Evaluates actual verified CPU features (AVX2, AVX-512, AVX-VNNI), available RAM,
 *    and operational GPU backends.
 * 3. Populates out_profile with the chosen profile type, readable name, and detailed reason.
 */
void arda_profile_evaluate(
    const struct arda_cpu_info *cpu,
    const struct arda_memory_info *mem,
    const struct arda_backend_status *backends,
    size_t backend_count,
    arda_profile_info_t *out_profile
);

/**
 * Returns a human-readable string identifier for a profile type.
 */
const char *arda_profile_type_to_string(arda_profile_type_t type);

#ifdef __cplusplus
}
#endif

#endif /* ARDA_PROFILE_H */
