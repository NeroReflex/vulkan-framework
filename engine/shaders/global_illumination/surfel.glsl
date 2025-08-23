#ifndef _SURFEL_
#define _SURFEL_

#include "../config.glsl"

#include "../morton.glsl"

#define SURFELS_FULL        0xFFFFFFFFu
#define SURFELS_MISSED      0xFFFFFFFEu
#define SURFELS_TOO_CLOSE   0xFFFFFFFDu

#define SURFEL_FLAG_LOCKED      (0x01u << 0u)
#define SURFEL_FLAG_PRIMARY     (0x01u << 1u)

struct Surfel {
    uint instance_id;

    float position_x;
    float position_y;
    float position_z;
    float radius;
    
    float normal_x;
    float normal_y;
    float normal_z;
    uint normal_samples;
    
    float irradiance_r;
    float irradiance_g;
    float irradiance_b;
    uint irradiance_samples;

    uint flags;

    uint morton;

    uint padding[1];
};

layout (set = 5, binding = 0, std430) /*coherent*/ buffer surfel_stats {
    uint total_surfels;
    uint free_surfels;
    uint busy_surfels;
    uint frame_spawned;
};

layout (set = 5, binding = 1, std430) /*coherent*/ buffer surfel_buffer_data { 
    Surfel surfels[];
};

bool can_spawn_another_surfel() {
    return atomicMax(frame_spawned, 0) < MAX_SURFELS_PER_FRAME;
}

bool is_point_in_surfel(uint surfel_id, const in vec3 point) {
    const vec3 center = vec3(surfels[surfel_id].position_x, surfels[surfel_id].position_y, surfels[surfel_id].position_z);
    const float radius = surfels[surfel_id].radius;

    const vec3 direction = point - center;

    return length(direction) <= radius;
    
    // this is supposed to be more efficient
    //return dot(direction, direction) <= radius * radius;
}

// Given the number of surfels already checked (to see if it would have been fitted into any of them),
// allocate a new surfel and return its index. If no surfel is available, return MAX_U32.
uint allocate_surfel(uint checked_surfels) {
    // check for free surfels left
    if (checked_surfels >= total_surfels) {
        return SURFELS_FULL;
    }

    uint prev_allocated = atomicCompSwap(busy_surfels, checked_surfels, checked_surfels + 1);

    return prev_allocated == checked_surfels ? prev_allocated : SURFELS_MISSED;
}

uint allocated_surfels() {
    return atomicMax(busy_surfels, 0);
}

bool lock_surfel(uint surfel_id) {
    return (atomicOr(surfels[surfel_id].flags, SURFEL_FLAG_LOCKED) & SURFEL_FLAG_LOCKED) == 0u;
}

void unlock_surfel(uint surfel_id) {
    atomicAnd(surfels[surfel_id].flags, ~SURFEL_FLAG_LOCKED);
}

void init_surfel(
    uint surfel_id,
    uint flags,
    uint instance_id,
    vec3 position,
    float radius,
    vec3 normal,
    vec3 irradiance,
    uint morton_code
) {
    atomicOr(surfels[surfel_id].flags, SURFEL_FLAG_LOCKED);

    surfels[surfel_id].instance_id        = instance_id;
    surfels[surfel_id].position_x         = position.x;
    surfels[surfel_id].position_y         = position.y;
    surfels[surfel_id].position_z         = position.z;
    surfels[surfel_id].radius             = radius;
    surfels[surfel_id].normal_x           = normal.x;
    surfels[surfel_id].normal_y           = normal.y;
    surfels[surfel_id].normal_z           = normal.z;
    surfels[surfel_id].normal_samples     = 1u;
    surfels[surfel_id].irradiance_r       = irradiance.r;
    surfels[surfel_id].irradiance_g       = irradiance.g;
    surfels[surfel_id].irradiance_b       = irradiance.b;
    surfels[surfel_id].irradiance_samples = 1u;
    surfels[surfel_id].morton             = morton_code;

    atomicAnd(surfels[surfel_id].flags, SURFEL_FLAG_LOCKED);
    atomicOr(surfels[surfel_id].flags, flags & ~SURFEL_FLAG_LOCKED);
}

uint linear_search_surfel_for_allocation(uint last_surfel_id, vec3 point, float radius) {
    bool too_close = false;
    for (uint i = 0; i < last_surfel_id; i++) {
        if ((is_point_in_surfel(i, point))) {
            return i;
        }

        if (distance(point, vec3(surfels[i].position_x, surfels[i].position_y, surfels[i].position_z)) < (radius + surfels[i].radius)) {
            too_close = true;
            // do not break, we want to check all surfels for matches,
            // but we also want to know if we were too close to any of them
            // to avoid allocating new ones
        }
    }

    return too_close ? SURFELS_TOO_CLOSE : SURFELS_MISSED;
}

uint linear_search_surfel(uint last_surfel_id, vec3 point, uint instance_id) {
    for (uint i = 0; i < last_surfel_id; i++) {
        if ((surfels[i].instance_id == instance_id) && (is_point_in_surfel(i, point))) {
            return i;
        }
    }

    return SURFELS_MISSED;
}

uint linear_search_surfel_ignore_instance_id(uint last_surfel_id, vec3 point) {
    for (uint i = 0; i < last_surfel_id; i++) {
        if (is_point_in_surfel(i, point)) {
            return i;
        }
    }

    return SURFELS_MISSED;
}

#define REGISTER_SURFEL_OK 0
#define REGISTER_SURFEL_FRAME_LIMIT 1
#define REGISTER_SURFEL_FULL 2
#define REGISTER_SURFEL_DENSITY 3
#define REGISTER_SURFEL_OUT_OF_RANGE 4;
#define REGISTER_SURFEL_BUSY 5;

uint register_surfel(
    in const vec3 eye_position,
    in const vec2 clip_planes,
    uint instance_id,
    bool primary,
    vec3 position,
    float radius,
    vec3 normal,
    vec3 irradiance
) {
    uint flags = primary ? SURFEL_FLAG_PRIMARY : 0u;

    uint morton = morton3D(eye_position, position, clip_planes);
    if (morton == MORTON_OUT_OF_SCALE) {
        return REGISTER_SURFEL_OUT_OF_RANGE;
    }

    bool done = false;
    uint checked_surfels = 0;
    do {
        // try to reuse an existing surfel
        checked_surfels = allocated_surfels();
        uint surfel_search_res = linear_search_surfel_for_allocation(checked_surfels, position, radius);
        if ((surfel_search_res != SURFELS_MISSED) && (surfel_search_res != SURFELS_TOO_CLOSE)) {
            //debugPrintfEXT("|REUSED(%u/%u)", surfel_search_res, checked_surfels);
            const uint surfel_id = surfel_search_res;

            // try locking the surfel
            if (!lock_surfel(surfel_id)) {
                // surfel is locked, avoid wasting time and just return busy
                return REGISTER_SURFEL_BUSY;
            }

            // surfel is locked: update it
            //const vec3 surfel_diffuse = vec3(surfels[surfel_search_res].irradiance_r, surfels[surfel_search_res].irradiance_g, surfels[surfel_search_res].irradiance_b) / float(surfels[surfel_search_res].irradiance_samples);
            //lights_contribution += contribution(surfel_diffuse, length(light_intensity), diffuse_contribution);

            // surfel is locked: unlock it for other instances
            unlock_surfel(surfel_id);
            memoryBarrierBuffer();

            return REGISTER_SURFEL_OK;
        } else if (surfel_search_res == SURFELS_TOO_CLOSE) {
            // we were too close to an existing surfel: do not allocate a new one
            //debugPrintfEXT("|TOO CLOSE");
            return REGISTER_SURFEL_DENSITY;
        } else if (!can_spawn_another_surfel()) {
            // we cannot allocate more surfels this frame
            // to avoid impacting too much on the frame time
            debugPrintfEXT("|FRAME_LIMIT");
            return REGISTER_SURFEL_FRAME_LIMIT;
        } else {
            // A matching surfel was not found: try to allocate a new one
            uint surfel_id = allocate_surfel(checked_surfels);
            if (surfel_id == SURFELS_FULL) {
                debugPrintfEXT("|FULL(%u)", surfel_id);
                return REGISTER_SURFEL_FULL;
            } else if (surfel_id == SURFELS_MISSED) {
                checked_surfels = allocated_surfels();
            } else {
                init_surfel(surfel_id, flags, instance_id, position, radius, normal, irradiance, morton);
                memoryBarrierBuffer();
                return REGISTER_SURFEL_OK;

                //debugPrintfEXT("\nCreated surfel %u at position vec3(%f, %f, %f)", surfel_id, surfels[surfel_id].position_x, surfels[surfel_id].position_y, surfels[surfel_id].position_z);
            }
        }
    } while (1 == 1);
}

bool surfel_is_primary(uint surfel_id) {
    return (surfels[surfel_id].flags & SURFEL_FLAG_PRIMARY) != 0u;
}

#endif // _SURFEL_
