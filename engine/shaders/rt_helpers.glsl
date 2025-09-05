#ifndef _RT_HELPERS_GLSL_
#define _RT_HELPERS_GLSL_

#ifndef STATUS_DESCRIPTOR_SET
    #error "STATUS_DESCRIPTOR_SET not defined"
#endif

#ifndef RT_DESCRIPTOR_SET
    #error "RT_DESCRIPTOR_SET not defined"
#endif

vec3 rt_directional_light(const uint surfel_id) {
    vec3 surfel_received_direction_light = vec3(0.0, 0.0, 0.0);
    for (uint light_index = 0; light_index < MAX_DIRECTIONAL_LIGHTS; light_index++) {
        const vec3 light_intensity = vec3(light[light_index].intensity_x, light[light_index].intensity_y, light[light_index].intensity_z);
        const vec3 light_dir = vec3(light[light_index].direction_x, light[light_index].direction_y, light[light_index].direction_z);
        const vec3 ray_dir = -1.0 * light_dir;
        if (length(light_dir) < 0.4) {
            continue;
        }

        // other flags: gl_RayFlagsCullNoOpaqueEXT gl_RayFlagsNoneEXT
        traceRayEXT(
            topLevelAS,
            gl_RayFlagsSkipAABBEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsCullNoOpaqueEXT,
            0xff,
            0,
            0,
            0,
            surfelPosition(surfel_id),
            CLOSEST_INTERSECTION_DISTANCE,
            ray_dir.xyz,
            10000.0,
            0
        );

        const float diffuse_contribution = max(dot(surfelNormal(surfel_id), ray_dir), 0.0);
        surfel_received_direction_light += diffuse_contribution * light_intensity;
    }

    return surfel_received_direction_light;
}

#endif //_RT_HELPERS_GLSL_