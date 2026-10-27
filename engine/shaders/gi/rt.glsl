#ifndef _GI_RT_
#define _GI_RT_

// Requires rt_descriptor_set.glsl (topLevelAS) and status.glsl (light[]).
#include "payload.glsl"
#include "common.glsl"

layout(location = 0) rayPayloadEXT hit_payload_t payload;

// True when nothing blocks the segment [origin, origin + dir * t_max].
bool trace_visible(vec3 origin, vec3 dir, float t_max) {
    payload.hit = 1u;
    traceRayEXT(
        topLevelAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsOpaqueEXT,
        0xff, 0, 0, 0,
        origin, RAY_T_MIN, dir, t_max,
        0
    );
    return payload.hit == 0u;
}

// Closest hit along the ray; fills `payload`. Returns false on a miss.
bool trace_closest(vec3 origin, vec3 dir) {
    payload.hit = 0u;
    traceRayEXT(
        topLevelAS,
        gl_RayFlagsOpaqueEXT,
        0xff, 0, 0, 0,
        origin, RAY_T_MIN, dir, RAY_T_MAX,
        0
    );
    return payload.hit != 0u;
}

// Closest hit within [RAY_T_MIN, t_max]; fills `payload`.
bool trace_closest_range(vec3 origin, vec3 dir, float t_max) {
    payload.hit = 0u;
    traceRayEXT(
        topLevelAS,
        gl_RayFlagsOpaqueEXT,
        0xff, 0, 0, 0,
        origin, RAY_T_MIN, dir, t_max,
        0
    );
    return payload.hit != 0u;
}

// Irradiance from the directional lights at a point with normal n.
// `origin` is the already offset ray origin. light[i].direction is the
// direction the light travels; the shadow ray goes the opposite way.
vec3 direct_irradiance(vec3 origin, vec3 n) {
    vec3 e = vec3(0.0);
    for (uint i = 0u; i < MAX_DIRECTIONAL_LIGHTS; i++) {
        vec3 travel = vec3(light[i].direction_x, light[i].direction_y, light[i].direction_z);
        vec3 intensity = vec3(light[i].intensity_x, light[i].intensity_y, light[i].intensity_z);
        float len = length(travel);
        if (len < 0.4 || max(intensity.r, max(intensity.g, intensity.b)) <= 0.0) {
            continue;
        }
        vec3 to_light = -travel / len;
        float c = dot(n, to_light);
        if (c <= 0.0) {
            continue;
        }
        if (trace_visible(origin, to_light, RAY_T_MAX)) {
            e += c * intensity;
        }
    }
    return e;
}

#endif // _GI_RT_
