#ifndef _DEBUG_
#define _DEBUG_

#ifndef DEBUG_DESCRIPTOR_SET
    #define DEBUG_DESCRIPTOR_SET 2
#endif

#define DEBUG_SHOW_SURFELS (1u << 0u)
#define DEBUG_APPLY_AO (1u << 1u)

// rust: ViewMode in rendering/debug_state.rs
#define VIEW_FINAL 0u
#define VIEW_SURFEL_IDS 1u
#define VIEW_COVERAGE 2u
#define VIEW_INDIRECT 3u
#define VIEW_INDIRECT_IRRADIANCE 4u
#define VIEW_REFERENCE 5u
#define VIEW_REFERENCE_INDIRECT 6u
#define VIEW_DIFFERENCE 7u
#define VIEW_AMBIENT_OCCLUSION 8u

// rust: AoAlgorithm
#define AO_SSAO 0u
#define AO_RTAO 1u

// rust: DebugGpuBlock (std140, 32 bytes)
layout(std140, set = DEBUG_DESCRIPTOR_SET, binding = 0) uniform debug_buffer
{
    uint flags;
    uint view_mode;
    float exposure;
    uint ao_algorithm;
    uvec4 pad1;
} debug_data;

#endif // _DEBUG_
