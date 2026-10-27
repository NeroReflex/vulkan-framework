#ifndef _GI_COMPUTE_
#define _GI_COMPUTE_

// Descriptor layout shared by every GI compute pass.
#extension GL_GOOGLE_include_directive : enable

#define STATUS_DESCRIPTOR_SET 0
#include "../status.glsl"

#define GBUFFER_DESCRIPTOR_SET 1
#include "../gbuffer.glsl"

#define SURFEL_SET 2
#include "surfels.glsl"

layout(push_constant) uniform GiPush {
    uint phase;
    uint arg0;
    uint arg1;
    uint arg2;
} pc;

#endif // _GI_COMPUTE_
