# vulkan_framework
vulkan_framework is my C++ vulkan-framework ported to Rust from C++ whit lots of improvements.

This framework aims to assist the developer with vulkan common operations and resources creation and memory allocation without limiting possibilities of what can be done.

You are __NOT__ limited to the functionality of the framework: while the C++ version exposed native vulkan handles this one exposes ash data that is a wrapper around vulkan raw API specification, therefore evey feature of vulkan can be used as it is usable in plain C.

## Portability
This project is designed to run everywhere vulkan 1.0.0 with no extensions is supported and a rust compiler is available!

To achieve this goal only rust's standard library and ash is used, everything else that might be needed (as the sdl2 glue) __MUST__ be manually imported by the developer.

## Memory Management
The framework makes it very clear what resources needs memory, and the user of the library is responsible for memory management for those resources.

The framework however assist the developer for this: resources requiring memory to be allocated have an handle that represents the occupied share of the
underlying memory so that when those resources goes out ot scope the memory is automatically freed, moreover it is not possible to create spece-requiring
resources without that handle, so you simply cannot forget to allocate memory for those resources!

All that is needed is to specify the memory allocator algorithm and what type of memory heap should it manage, but writing a memory allocator is a daunting task,
so the framework will come equipped with a few default implementations!

