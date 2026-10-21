#include "arda/arda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *prog_name) {
    printf("Arda's Breath Hardware Capability Probe (v%s)\n\n", ARDA_VERSION_STRING);
    printf("Usage: %s [OPTIONS]\n\n", prog_name);
    printf("Options:\n");
    printf("  --text               Print human-readable capability report (default)\n");
    printf("  --json               Print report in JSON format\n");
    printf("  --from-proc <C> <M>  Probe using custom /proc/cpuinfo <C> and /proc/meminfo <M> paths\n");
    printf("  --check              Verify probe runs cleanly and print summary\n");
    printf("  -h, --help           Display this help message and exit\n");
    printf("  -v, --version        Display version information and exit\n");
}

int main(int argc, char **argv) {
    bool json_mode = false;
    bool check_mode = false;
    const char *custom_cpuinfo = NULL;
    const char *custom_meminfo = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--version") == 0 || strcmp(argv[i], "-v") == 0) {
            printf("arda-probe %s\n", ARDA_VERSION_STRING);
            return 0;
        } else if (strcmp(argv[i], "--json") == 0) {
            json_mode = true;
        } else if (strcmp(argv[i], "--text") == 0) {
            json_mode = false;
        } else if (strcmp(argv[i], "--check") == 0) {
            check_mode = true;
        } else if (strcmp(argv[i], "--from-proc") == 0) {
            if (i + 2 < argc) {
                custom_cpuinfo = argv[++i];
                custom_meminfo = argv[++i];
            } else {
                fprintf(stderr, "Error: --from-proc requires two arguments: <cpuinfo_path> <meminfo_path>\n");
                return 1;
            }
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    arda_capability_report_t report;
    int rc = 0;

    if (custom_cpuinfo || custom_meminfo) {
        rc = arda_capability_report_collect_from_proc(&report, custom_cpuinfo, custom_meminfo);
    } else {
        rc = arda_capability_report_collect(&report);
    }

    if (rc != 0) {
        fprintf(stderr, "Error: Failed to collect system capability report\n");
        return 1;
    }

    if (check_mode) {
        printf("Capability check passed: CPU=%s (%u cores), RAM=%.1f GB, Profile=%s\n",
               report.cpu.model_name,
               report.cpu.logical_cores,
               (double)report.memory.total_ram_bytes / (1024.0 * 1024.0 * 1024.0),
               report.selected_profile.name);
        return 0;
    }

    char output_buf[16384];
    if (json_mode) {
        if (arda_capability_report_format_json(&report, output_buf, sizeof(output_buf)) != 0) {
            fprintf(stderr, "Error: Output buffer too small for JSON report\n");
            return 1;
        }
    } else {
        if (arda_capability_report_format_text(&report, output_buf, sizeof(output_buf)) != 0) {
            fprintf(stderr, "Error: Output buffer too small for text report\n");
            return 1;
        }
    }

    printf("%s", output_buf);
    return 0;
}
