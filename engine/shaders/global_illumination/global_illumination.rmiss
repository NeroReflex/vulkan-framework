//#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "payload.glsl"

layout(location = 0) rayPayloadEXT hit_payload_t payload;

void main() {
    payload.hit = false;
}
