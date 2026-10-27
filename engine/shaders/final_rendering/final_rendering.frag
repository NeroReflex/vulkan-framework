#include "../debug.glsl"

layout (location = 0) out vec4 outColor;

layout (location = 0) in vec2 in_vTextureUV;

#define GBUFFER_DESCRIPTOR_SET 0
#include "../gbuffer.glsl"

// dominant surfel id per pixel (0xFFFFFFFF = none)
layout(set = 1, binding = 0) uniform usampler2D surfels_buffer;

// rust: GI_SAMPLED_* in global_illumination.rs
#define GI_INDIRECT 0
#define GI_DIRECT 1
#define GI_REFERENCE 2
#define GI_COVERAGE 3
layout(set = 1, binding = 1) uniform sampler2D gibuffer[4];

float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec3 id_color(uint id) {
    uint h = id * 0x9E3779B1u;
    h ^= h >> 15u;
    h *= 0x85EBCA77u;
    h ^= h >> 13u;
    return vec3(float(h & 0xFFu), float((h >> 8u) & 0xFFu), float((h >> 16u) & 0xFFu)) / 255.0;
}

void main() {
    const ivec2 texel = ivec2(gl_FragCoord.xy);
    const vec3 albedo = texelFetch(gbuffer[2], texel, 0).rgb;
    const bool sky = texelFetch(gbuffer_instance_id, texel, 0).x == 0xFFFFFFFFu;
    const vec3 e_ind = texelFetch(gibuffer[GI_INDIRECT], texel, 0).rgb;
    const vec3 e_dir = texelFetch(gibuffer[GI_DIRECT], texel, 0).rgb;
    const vec3 e_ref = texelFetch(gibuffer[GI_REFERENCE], texel, 0).rgb;
    const vec4 coverage = texelFetch(gibuffer[GI_COVERAGE], texel, 0);

    vec3 color;
    switch (debug_data.view_mode) {
    case VIEW_SURFEL_IDS: {
        const uint id = texelFetch(surfels_buffer, texel, 0).x;
        const vec3 shade = albedo * (e_dir + e_ind);
        color = id == 0xFFFFFFFFu ? shade : mix(shade, id_color(id) * max(luminance(shade), 0.25), 0.6);
        break;
    }
    case VIEW_COVERAGE: {
        const float c = coverage.x;
        if (c < 0.5) {
            color = mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0), c / 0.5);
        } else if (c < 1.5) {
            color = vec3(0.0, 1.0, 0.0) * (0.5 + 0.5 * (c - 0.5));
        } else {
            color = mix(vec3(0.0, 1.0, 0.0), vec3(0.0, 0.3, 1.0), clamp((c - 1.5) / 2.5, 0.0, 1.0));
        }
        color *= 0.25 + 0.75 * clamp(luminance(albedo) * 2.0, 0.0, 1.0);
        break;
    }
    case VIEW_INDIRECT:
        color = albedo * e_ind;
        break;
    case VIEW_INDIRECT_IRRADIANCE:
        color = e_ind;
        break;
    case VIEW_REFERENCE:
        color = albedo * (e_dir + e_ref);
        break;
    case VIEW_REFERENCE_INDIRECT:
        color = albedo * e_ref;
        break;
    case VIEW_DIFFERENCE: {
        const float a = luminance(e_ind);
        const float b = luminance(e_ref);
        const float rel = (a - b) / max(b, 1e-3 + 0.05 * max(a, b));
        color = rel > 0.0 ? vec3(clamp(rel, 0.0, 1.0), 0.0, 0.0) : vec3(0.0, 0.0, clamp(-rel, 0.0, 1.0));
        color = mix(color, vec3(0.08), 1.0 - clamp(abs(rel) * 4.0, 0.0, 1.0));
        break;
    }
    default:
        color = albedo * (e_dir + e_ind);
        break;
    }
    if (sky && debug_data.view_mode != VIEW_COVERAGE) {
        color = vec3(0.0);
    }

    outColor = vec4(color * debug_data.exposure, 1.0);
}
