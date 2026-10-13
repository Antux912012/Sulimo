# Third-Party Notices

Arda's Breath is designed to integrate or link with established third-party components.
In accordance with the project engineering plan, this document records all third-party
dependencies, their origins, and their license terms.

---

### Planned Dependencies

1. **llama.cpp**
   - **Origin:** https://github.com/ggerganov/llama.cpp
   - **License:** MIT License
   - **Usage:** Upstream inference engine backend for GGUF model loading, CPU kernels,
     and optional GPU backends (Phase 1+).
   - **Notice:** Copyright (c) 2023-2026 Georgi Gerganov and llama.cpp contributors.

2. **Khronos Group (Vulkan & OpenCL Headers / Loaders)**
   - **Origin:** https://github.com/KhronosGroup
   - **License:** Apache License 2.0 / MIT
   - **Usage:** Standard graphics and compute API definitions for runtime hardware probing and acceleration.

3. **GNOME Libadwaita & GTK4**
   - **Origin:** https://gitlab.gnome.org/GNOME/libadwaita, https://gitlab.gnome.org/GNOME/gtk
   - **License:** GNU Lesser General Public License (LGPL) v2.1+
   - **Usage:** Desktop client user interface (Phase 5).
