#include "arda/capability.h"
#include "../test_helpers.h"

static void test_live_system_probe(void) {
    arda_capability_report_t report;
    int rc = arda_capability_report_collect(&report);
    ASSERT_EQ_INT(rc, 0);

    /* Verify OS */
    ASSERT_TRUE(report.os.sysname[0] != '\0');
    ASSERT_TRUE(report.os.machine[0] != '\0');

    /* Verify CPU */
    ASSERT_TRUE(report.cpu.logical_cores >= 1);
    ASSERT_TRUE(report.cpu.model_name[0] != '\0');
    ASSERT_TRUE(report.cpu.vendor[0] != '\0');

    /* Verify Memory */
    ASSERT_TRUE(report.memory.total_ram_bytes > 0);
    ASSERT_TRUE(report.memory.available_ram_bytes > 0);
    ASSERT_TRUE(report.memory.page_size_bytes >= 4096);

    /* Verify Backends */
    ASSERT_TRUE(report.backend_count >= 1);
    ASSERT_EQ_INT(report.backends[0].id, ARDA_BACKEND_CPU);
    ASSERT_TRUE(report.backends[0].compiled_in);
    ASSERT_TRUE(report.backends[0].runtime_available);

    /* Verify Profile selection */
    ASSERT_TRUE(report.selected_profile.type != ARDA_PROFILE_UNKNOWN);
    ASSERT_TRUE(report.selected_profile.name != NULL);
    ASSERT_TRUE(report.selected_profile.reason[0] != '\0');

    /* Verify text format output */
    char text_buf[16384];
    int fmt_rc = arda_capability_report_format_text(&report, text_buf, sizeof(text_buf));
    ASSERT_EQ_INT(fmt_rc, 0);
    ASSERT_TRUE(strlen(text_buf) > 100);

    /* Verify JSON format output */
    char json_buf[16384];
    int json_rc = arda_capability_report_format_json(&report, json_buf, sizeof(json_buf));
    ASSERT_EQ_INT(json_rc, 0);
    ASSERT_TRUE(strlen(json_buf) > 100);
}

int main(void) {
    printf("[Unit Test] Live System Hardware Capability Probe\n");
    RUN_TEST(test_live_system_probe);
    TEST_REPORT_MAIN();
}
