#include "arda/capability.h"
#include "../test_helpers.h"

static void init_mock_report(arda_capability_report_t *rep) {
    memset(rep, 0, sizeof(*rep));

    strncpy(rep->os.sysname, "Linux", sizeof(rep->os.sysname) - 1);
    strncpy(rep->os.release, "6.1.0", sizeof(rep->os.release) - 1);
    strncpy(rep->os.machine, "x86_64", sizeof(rep->os.machine) - 1);

    strncpy(rep->cpu.vendor, "GenuineIntel", sizeof(rep->cpu.vendor) - 1);
    strncpy(rep->cpu.model_name, "Mock Intel i5", sizeof(rep->cpu.model_name) - 1);
    rep->cpu.logical_cores = 4;
    rep->cpu.physical_cores = 2;
    rep->cpu.has_avx = true;
    rep->cpu.has_avx2 = true;

    rep->memory.total_ram_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
    rep->memory.available_ram_bytes = 10ULL * 1024ULL * 1024ULL * 1024ULL;
    rep->memory.page_size_bytes = 4096;

    rep->backend_count = 1;
    rep->backends[0].id = ARDA_BACKEND_CPU;
    rep->backends[0].name = "CPU";
    rep->backends[0].compiled_in = true;
    rep->backends[0].runtime_available = true;

    rep->selected_profile.type = ARDA_PROFILE_CLASSIC;
    rep->selected_profile.name = "arda-classic";
    strncpy(rep->selected_profile.reason, "AVX2 detected", sizeof(rep->selected_profile.reason) - 1);
}

static void test_text_formatting(void) {
    arda_capability_report_t rep;
    init_mock_report(&rep);

    char buf[8192];
    int rc = arda_capability_report_format_text(&rep, buf, sizeof(buf));
    ASSERT_EQ_INT(rc, 0);

    ASSERT_STR_CONTAINS(buf, "ARDA'S BREATH - CAPABILITY REPORT");
    ASSERT_STR_CONTAINS(buf, "[Operating System]");
    ASSERT_STR_CONTAINS(buf, "Linux 6.1.0");
    ASSERT_STR_CONTAINS(buf, "[CPU Architecture & Features]");
    ASSERT_STR_CONTAINS(buf, "Mock Intel i5");
    ASSERT_STR_CONTAINS(buf, "AVX2=yes");
    ASSERT_STR_CONTAINS(buf, "[System Memory]");
    ASSERT_STR_CONTAINS(buf, "[Inference & Acceleration Backends]");
    ASSERT_STR_CONTAINS(buf, "arda-classic");
}

static void test_json_formatting(void) {
    arda_capability_report_t rep;
    init_mock_report(&rep);

    char buf[8192];
    int rc = arda_capability_report_format_json(&rep, buf, sizeof(buf));
    ASSERT_EQ_INT(rc, 0);

    ASSERT_STR_CONTAINS(buf, "\"os\": {");
    ASSERT_STR_CONTAINS(buf, "\"sysname\": \"Linux\"");
    ASSERT_STR_CONTAINS(buf, "\"cpu\": {");
    ASSERT_STR_CONTAINS(buf, "\"model_name\": \"Mock Intel i5\"");
    ASSERT_STR_CONTAINS(buf, "\"memory\": {");
    ASSERT_STR_CONTAINS(buf, "\"total_ram_bytes\":");
    ASSERT_STR_CONTAINS(buf, "\"backends\": [");
    ASSERT_STR_CONTAINS(buf, "\"profile\": {");
    ASSERT_STR_CONTAINS(buf, "\"name\": \"arda-classic\"");
}

static void test_small_buffer_protection(void) {
    arda_capability_report_t rep;
    init_mock_report(&rep);

    char tiny[32];
    int rc_text = arda_capability_report_format_text(&rep, tiny, sizeof(tiny));
    ASSERT_EQ_INT(rc_text, -1);

    int rc_json = arda_capability_report_format_json(&rep, tiny, sizeof(tiny));
    ASSERT_EQ_INT(rc_json, -1);
}

int main(void) {
    printf("[Unit Test] Capability Report Formatting (Text & JSON)\n");
    RUN_TEST(test_text_formatting);
    RUN_TEST(test_json_formatting);
    RUN_TEST(test_small_buffer_protection);
    TEST_REPORT_MAIN();
}
