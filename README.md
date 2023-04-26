# vulkan_framework
vulkan_framework is my C++ vulkan-framework ported to Rust from C++ whit lots of improvements.

This framework aims to assist the developer with vulkan common operations and resources creation and memory allocation without limiting possibilities of what can be done.

You are __NOT__ limited to the functionality of the framework: while the C++ version exposed native vulkan handles this one exposes ash data that is a wrapper around vulkan raw API specification, therefore evey feature of vulkan can be used as it is usable in plain C.

## Portability
This project is designed to run everywhere vulkan 1.0.0 with no extensions is supported and a rust compiler is available!

To achieve this goal only rust's standard library and ash is used, everything else that might be needed (as the sdl2 glue) __MUST__ be manually imported by the developer.

Moreover everything that depends on an extension is optional and is not "flattened" as it is in the vulkan documentation, insted it is very explicit when you are using a Vulkan extension!

## Memory Management
The framework makes it very clear what resources needs memory, and the user of the library is responsible for memory management for those resources.

The framework however assist the developer for this: resources requiring memory to be allocated have an handle that represents the occupied share of the
underlying memory so that when those resources goes out ot scope the memory is automatically freed, moreover it is not possible to create spece-requiring
resources without that handle, so you simply cannot forget to allocate memory for those resources!

All that is needed is to specify the memory allocator algorithm and what type of memory heap should it manage, but writing a memory allocator is a daunting task,
so the framework will come equipped with a few default implementations!

As for Sparse memory binding that is planned to be supported in a future release.

## Raytracing
In spite of aiming to basic vulkan 1.0 compatibility, for this framework extensions *VK_KHR_ray_tracing_pipeline* and *VK_KHR_ray_query* are first-class citizens
and support for ray-tracing pipeline is a focal point.

## Resource Tracking at GPU side
This framework gives the developer basic tools to track resource usage on the GPU so that it's impossible to destroy handles of resources while those are used.

This functionality will be the main development point after the first release.

This functionality does not place automatic barriers, you are still responsible for every aspect of GPU and Host memory synchronization! You are a vulkan developer, after all, that is what we want! Say no to OpenGL!

The developer can obviously just ignore this feature and in that case CPU <-> GPU coherency is up to him.