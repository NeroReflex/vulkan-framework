//#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

#extension GL_GOOGLE_include_directive : enable


#include "payload.glsl"

#define RT_DESCRIPTOR_SET 0
#include "../rt_descriptor_set.glsl"

#define TEXTURES_DESCRIPTOR_SET 3
#include "../textures.glsl"

#define MATERIALS_DESCRIPTOR_SET 4
#include "../materials.glsl"

layout(location = 0) rayPayloadInEXT hit_payload_t payload;

hitAttributeEXT vec2 attribs;

void main() {
    payload.hit = true;
}
