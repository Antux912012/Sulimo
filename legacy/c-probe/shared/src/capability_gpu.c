#include "arda/capability.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>

/* Minimal portable Vulkan definitions to probe without requiring vulkan-dev headers */
#define VK_SUCCESS 0
#define VK_STRUCTURE_TYPE_APPLICATION_INFO 0
#define VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO 1
#define VK_MAKE_API_VERSION(variant, major, minor, patch) \
    ((((uint32_t)(variant)) << 29U) | (((uint32_t)(major)) << 22U) | (((uint32_t)(minor)) << 12U) | ((uint32_t)(patch)))

#define VK_PHYSICAL_DEVICE_TYPE_OTHER 0
#define VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU 1
#define VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU 2
#define VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU 3
#define VK_PHYSICAL_DEVICE_TYPE_CPU 4
#define VK_MEMORY_HEAP_DEVICE_LOCAL_BIT 0x00000001

typedef void* VkInstance;
typedef void* VkPhysicalDevice;

typedef struct {
    int sType;
    const void *pNext;
    const char *pApplicationName;
    uint32_t applicationVersion;
    const char *pEngineName;
    uint32_t engineVersion;
    uint32_t apiVersion;
} VkApplicationInfo_mini;

typedef struct {
    int sType;
    const void *pNext;
    uint32_t flags;
    const VkApplicationInfo_mini *pApplicationInfo;
    uint32_t enabledLayerCount;
    const char *const *ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char *const *ppEnabledExtensionNames;
} VkInstanceCreateInfo_mini;

typedef struct {
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t deviceType;
    char deviceName[256];
    uint8_t pipelineCacheUUID[16];
    uint8_t limits_and_sparse[1024]; /* buffer for remaining property fields */
} VkPhysicalDeviceProperties_mini;

typedef struct {
    uint64_t size;
    uint32_t flags;
} VkMemoryHeap_mini;

typedef struct {
    uint32_t propertyFlags;
    uint32_t heapIndex;
} VkMemoryType_mini;

typedef struct {
    uint32_t memoryTypeCount;
    VkMemoryType_mini memoryTypes[32];
    uint32_t memoryHeapCount;
    VkMemoryHeap_mini memoryHeaps[16];
} VkPhysicalDeviceMemoryProperties_mini;

typedef int (*PFN_vkCreateInstance)(const VkInstanceCreateInfo_mini*, const void*, VkInstance*);
typedef void (*PFN_vkDestroyInstance)(VkInstance, const void*);
typedef int (*PFN_vkEnumeratePhysicalDevices)(VkInstance, uint32_t*, VkPhysicalDevice*);
typedef void (*PFN_vkGetPhysicalDeviceProperties)(VkPhysicalDevice, VkPhysicalDeviceProperties_mini*);
typedef void (*PFN_vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties_mini*);

/* Minimal OpenCL definitions */
#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_VENDOR 0x102C
#define CL_DRIVER_VERSION 0x102D
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F

typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef int cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;

typedef cl_int (*PFN_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*PFN_clGetDeviceIDs)(cl_platform_id, uint64_t, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (*PFN_clGetDeviceInfo)(cl_device_id, uint32_t, size_t, void*, size_t*);

static const char *vendor_id_to_name(uint32_t vendor_id) {
    switch (vendor_id) {
        case 0x8086: return "Intel";
        case 0x10de: return "NVIDIA";
        case 0x1002: return "AMD";
        case 0x13b5: return "ARM";
        case 0x5143: return "Qualcomm";
        case 0x1010: return "ImgTec";
        default:     return "Unknown";
    }
}

static void probe_drm_devices(void) {
    DIR *d = opendir("/sys/class/drm");
    if (d) {
        closedir(d);
    }
}

static void probe_vulkan_backend(arda_backend_status_t *backend) {
    backend->id = ARDA_BACKEND_VULKAN;
    backend->name = "Vulkan";
#ifdef ARDA_BACKEND_VULKAN_ENABLED
    backend->compiled_in = true;
#else
    backend->compiled_in = false;
#endif

    void *lib = dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        backend->runtime_available = false;
        const char *err = dlerror();
        snprintf(backend->error_reason, sizeof(backend->error_reason),
                 "Vulkan loader not found (%s)", err ? err : "dlopen failed");
        return;
    }

    PFN_vkCreateInstance pvkCreateInstance = NULL;
    PFN_vkDestroyInstance pvkDestroyInstance = NULL;
    PFN_vkEnumeratePhysicalDevices pvkEnumeratePhysicalDevices = NULL;
    PFN_vkGetPhysicalDeviceProperties pvkGetPhysicalDeviceProperties = NULL;
    PFN_vkGetPhysicalDeviceMemoryProperties pvkGetPhysicalDeviceMemoryProperties = NULL;

    *(void **)(&pvkCreateInstance) = dlsym(lib, "vkCreateInstance");
    *(void **)(&pvkDestroyInstance) = dlsym(lib, "vkDestroyInstance");
    *(void **)(&pvkEnumeratePhysicalDevices) = dlsym(lib, "vkEnumeratePhysicalDevices");
    *(void **)(&pvkGetPhysicalDeviceProperties) = dlsym(lib, "vkGetPhysicalDeviceProperties");
    *(void **)(&pvkGetPhysicalDeviceMemoryProperties) = dlsym(lib, "vkGetPhysicalDeviceMemoryProperties");

    if (!pvkCreateInstance || !pvkDestroyInstance || !pvkEnumeratePhysicalDevices ||
        !pvkGetPhysicalDeviceProperties || !pvkGetPhysicalDeviceMemoryProperties) {
        backend->runtime_available = false;
        snprintf(backend->error_reason, sizeof(backend->error_reason),
                 "%s", "Incomplete Vulkan entry points in libvulkan.so.1");
        dlclose(lib);
        return;
    }

    VkApplicationInfo_mini app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = NULL,
        .pApplicationName = "arda-probe",
        .applicationVersion = 1,
        .pEngineName = "arda",
        .engineVersion = 1,
        .apiVersion = VK_MAKE_API_VERSION(0, 1, 0, 0)
    };

    VkInstanceCreateInfo_mini create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = NULL,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = NULL
    };

    VkInstance instance = NULL;
    int res = pvkCreateInstance(&create_info, NULL, &instance);
    if (res != VK_SUCCESS || !instance) {
        backend->runtime_available = false;
        snprintf(backend->error_reason, sizeof(backend->error_reason),
                 "vkCreateInstance failed (error code %d)", res);
        dlclose(lib);
        return;
    }

    uint32_t dev_count = 0;
    pvkEnumeratePhysicalDevices(instance, &dev_count, NULL);
    if (dev_count == 0) {
        backend->runtime_available = false;
        snprintf(backend->error_reason, sizeof(backend->error_reason),
                 "%s", "No Vulkan physical devices reported by ICD");
        pvkDestroyInstance(instance, NULL);
        dlclose(lib);
        return;
    }

    VkPhysicalDevice *devices = (VkPhysicalDevice*)calloc(dev_count, sizeof(VkPhysicalDevice));
    if (!devices) {
        backend->runtime_available = false;
        snprintf(backend->error_reason, sizeof(backend->error_reason),
                 "%s", "Memory allocation failure during device query");
        pvkDestroyInstance(instance, NULL);
        dlclose(lib);
        return;
    }

    pvkEnumeratePhysicalDevices(instance, &dev_count, devices);
    backend->runtime_available = true;
    backend->error_reason[0] = '\0';

    uint32_t record_count = (dev_count < ARDA_MAX_GPU_DEVICES) ? dev_count : ARDA_MAX_GPU_DEVICES;
    backend->device_count = record_count;

    for (uint32_t i = 0; i < record_count; i++) {
        VkPhysicalDeviceProperties_mini props;
        memset(&props, 0, sizeof(props));
        pvkGetPhysicalDeviceProperties(devices[i], &props);

        VkPhysicalDeviceMemoryProperties_mini mem_props;
        memset(&mem_props, 0, sizeof(mem_props));
        pvkGetPhysicalDeviceMemoryProperties(devices[i], &mem_props);

        arda_gpu_device_t *gpu = &backend->devices[i];
        snprintf(gpu->name, sizeof(gpu->name), "%s", props.deviceName);
        snprintf(gpu->vendor, sizeof(gpu->vendor), "%s", vendor_id_to_name(props.vendorID));
        snprintf(gpu->driver_version, sizeof(gpu->driver_version), "%u.%u.%u",
                 (props.driverVersion >> 22) & 0x3ff,
                 (props.driverVersion >> 12) & 0x3ff,
                 props.driverVersion & 0xfff);

        gpu->is_integrated = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
        gpu->is_discrete   = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
        gpu->is_software   = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU);

        uint64_t total_vram = 0;
        for (uint32_t h = 0; h < mem_props.memoryHeapCount; h++) {
            if (mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                total_vram += mem_props.memoryHeaps[h].size;
            }
        }
        gpu->total_memory_bytes = total_vram;
    }

    free(devices);
    pvkDestroyInstance(instance, NULL);
    dlclose(lib);
}

static void probe_opencl_backend(arda_backend_status_t *backend) {
    backend->id = ARDA_BACKEND_OPENCL;
    backend->name = "OpenCL";
#ifdef ARDA_BACKEND_OPENCL_ENABLED
    backend->compiled_in = true;
#else
    backend->compiled_in = false;
#endif

    void *lib = dlopen("libOpenCL.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        backend->runtime_available = false;
        const char *err = dlerror();
        snprintf(backend->error_reason, sizeof(backend->error_reason),
                 "OpenCL loader not found (%s)", err ? err : "dlopen failed");
        return;
    }

    PFN_clGetPlatformIDs pclGetPlatformIDs = NULL;
    PFN_clGetDeviceIDs pclGetDeviceIDs = NULL;
    PFN_clGetDeviceInfo pclGetDeviceInfo = NULL;

    *(void **)(&pclGetPlatformIDs) = dlsym(lib, "clGetPlatformIDs");
    *(void **)(&pclGetDeviceIDs) = dlsym(lib, "clGetDeviceIDs");
    *(void **)(&pclGetDeviceInfo) = dlsym(lib, "clGetDeviceInfo");

    if (!pclGetPlatformIDs || !pclGetDeviceIDs || !pclGetDeviceInfo) {
        backend->runtime_available = false;
        snprintf(backend->error_reason, sizeof(backend->error_reason),
                 "%s", "Incomplete OpenCL entry points in libOpenCL.so.1");
        dlclose(lib);
        return;
    }

    cl_uint num_platforms = 0;
    cl_int err = pclGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        backend->runtime_available = false;
        snprintf(backend->error_reason, sizeof(backend->error_reason),
                 "%s", "No OpenCL platforms or ICDs installed");
        dlclose(lib);
        return;
    }

    cl_platform_id *platforms = (cl_platform_id*)calloc(num_platforms, sizeof(cl_platform_id));
    if (!platforms) {
        backend->runtime_available = false;
        snprintf(backend->error_reason, sizeof(backend->error_reason),
                 "%s", "Memory allocation failure querying OpenCL platforms");
        dlclose(lib);
        return;
    }
    pclGetPlatformIDs(num_platforms, platforms, NULL);

    uint32_t total_devices = 0;
    for (cl_uint p = 0; p < num_platforms && total_devices < ARDA_MAX_GPU_DEVICES; p++) {
        cl_uint num_devices = 0;
        if (pclGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices) == CL_SUCCESS && num_devices > 0) {
            cl_device_id *devices = (cl_device_id*)calloc(num_devices, sizeof(cl_device_id));
            if (devices) {
                pclGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
                for (cl_uint d = 0; d < num_devices && total_devices < ARDA_MAX_GPU_DEVICES; d++) {
                    arda_gpu_device_t *gpu = &backend->devices[total_devices];
                    pclGetDeviceInfo(devices[d], CL_DEVICE_NAME, sizeof(gpu->name), gpu->name, NULL);
                    pclGetDeviceInfo(devices[d], CL_DEVICE_VENDOR, sizeof(gpu->vendor), gpu->vendor, NULL);
                    pclGetDeviceInfo(devices[d], CL_DRIVER_VERSION, sizeof(gpu->driver_version), gpu->driver_version, NULL);
                    cl_ulong mem_size = 0;
                    pclGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
                    gpu->total_memory_bytes = (uint64_t)mem_size;
                    total_devices++;
                }
                free(devices);
            }
        }
    }
    free(platforms);

    if (total_devices > 0) {
        backend->runtime_available = true;
        backend->device_count = total_devices;
        backend->error_reason[0] = '\0';
    } else {
        backend->runtime_available = false;
        snprintf(backend->error_reason, sizeof(backend->error_reason),
                 "%s", "No OpenCL devices found for installed platforms");
    }

    dlclose(lib);
}

void arda_backends_probe(arda_backend_status_t *out_backends, size_t max_backends, size_t *out_count) {
    if (!out_backends || !out_count || max_backends < 4) return;
    memset(out_backends, 0, sizeof(*out_backends) * max_backends);

    probe_drm_devices();

    /* 1. CPU Backend */
    out_backends[0].id = ARDA_BACKEND_CPU;
    out_backends[0].name = "CPU";
    out_backends[0].compiled_in = true;
    out_backends[0].runtime_available = true;
    out_backends[0].error_reason[0] = '\0';
    out_backends[0].device_count = 1;
    snprintf(out_backends[0].devices[0].name, sizeof(out_backends[0].devices[0].name), "%s", "Host CPU");

    /* 2. Vulkan Backend */
    probe_vulkan_backend(&out_backends[1]);

    /* 3. OpenCL Backend */
    probe_opencl_backend(&out_backends[2]);

    /* 4. SYCL Backend */
    out_backends[3].id = ARDA_BACKEND_SYCL;
    out_backends[3].name = "Intel SYCL";
#ifdef ARDA_BACKEND_SYCL_ENABLED
    out_backends[3].compiled_in = true;
    out_backends[3].runtime_available = false;
    snprintf(out_backends[3].error_reason, sizeof(out_backends[3].error_reason),
             "%s", "SYCL runtime check requires oneAPI environment");
#else
    out_backends[3].compiled_in = false;
    out_backends[3].runtime_available = false;
    snprintf(out_backends[3].error_reason, sizeof(out_backends[3].error_reason),
             "%s", "Backend not enabled at compile time");
#endif

    *out_count = 4;
}
