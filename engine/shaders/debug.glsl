#ifndef _DEBUG_
#define _DEBUG_

#ifndef DEBUG_DESCRIPTOR_SET
    #define DEBUG_DESCRIPTOR_SET 2
#endif

#define DEBUG_SHOW_SURFELS (1u << 0u)

layout(std140, set = DEBUG_DESCRIPTOR_SET, binding = 0) uniform debug_buffer
{
    uint flags;
    uint padding[7];
} debug_data;

#endif // _DEBUG_
