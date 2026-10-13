#include "arda/capability.h"
#include <stdio.h>
#include <string.h>
#include <sys/utsname.h>

void arda_os_probe(arda_os_info_t *out_os) {
    if (!out_os) return;
    memset(out_os, 0, sizeof(*out_os));

    struct utsname u;
    if (uname(&u) == 0) {
        snprintf(out_os->sysname, sizeof(out_os->sysname), "%s", u.sysname);
        snprintf(out_os->release, sizeof(out_os->release), "%s", u.release);
        snprintf(out_os->version, sizeof(out_os->version), "%s", u.version);
        snprintf(out_os->machine, sizeof(out_os->machine), "%s", u.machine);
    } else {
        snprintf(out_os->sysname, sizeof(out_os->sysname), "%s", "Unknown");
        snprintf(out_os->machine, sizeof(out_os->machine), "%s", "Unknown");
    }
}
