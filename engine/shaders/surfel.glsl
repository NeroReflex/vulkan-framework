#ifndef _SURFEL_
#define _SURFEL_

#include "config.glsl"

#include "morton.glsl"

#ifndef SURFELS_DESCRIPTOR_SET
#define SURFELS_DESCRIPTOR_SET 5
#endif

uniform layout (set = SURFELS_DESCRIPTOR_SET, binding = 2, rgba32f) image2D outputImage[2];

#define SURFELS_FULL        0xFFFFFFFFu
#define SURFELS_MISSED      0xFFFFFFFEu
#define SURFELS_TOO_CLOSE   0xFFFFFFFDu

#define SURFEL_FLAG_LOCKED      (0x01u << 0u)
#define SURFEL_FLAG_PRIMARY     (0x01u << 1u)

#define RADIANCE_THRESHOLD 10.0f

struct Surfel {
    uint instance_id;

    float position_x;
    float position_y;
    float position_z;
    float radius;
    
    float normal_x;
    float normal_y;
    float normal_z;
    
    float irradiance_r;
    float irradiance_g;
    float irradiance_b;

    uint contributions;

    uint flags;

    uint morton;

    uint padding[1];
};

layout (set = SURFELS_DESCRIPTOR_SET, binding = 0, std430) /*coherent*/ buffer surfel_stats {
    int total_surfels;
    int unordered_surfels;
    int ordered_surfels;
};

layout (set = SURFELS_DESCRIPTOR_SET, binding = 1, std430) /*coherent*/ buffer surfel_buffer_data { 
    Surfel surfels[];
};

bool is_point_in_surfel(uint surfel_id, const in vec3 point) {
    const vec3 center = vec3(surfels[surfel_id].position_x, surfels[surfel_id].position_y, surfels[surfel_id].position_z);
    const float radius = surfels[surfel_id].radius;

    const vec3 direction = point - center;

    //return length(direction) <= radius;
    
    // this is supposed to be more efficient
    return dot(direction, direction) <= radius * radius;
}

uint count_ordered_surfels() {
    // ordered_surfels is never modified in raytrace shader,
    // so avoid the expensive atomic read
    return ordered_surfels;
}

uint count_unordered_surfels() {
    return atomicMax(unordered_surfels, 0);
}

// Given the number of UNORDERED surfels already checked (to see if it would have been fitted into any of them),
// allocate a new surfel and return its index. If no surfel is available, return MAX_U32.
uint allocate_surfel(uint checked_surfels) {
    const int scanned = int(checked_surfels);

    // check for free surfels left (leave the top-half untouched for reordering)
    if ((count_ordered_surfels() + checked_surfels) >= (total_surfels / 2)) {
        return SURFELS_FULL;
    }

    uint prev_allocated = atomicCompSwap(unordered_surfels, scanned, scanned + 1);

    return prev_allocated == checked_surfels ? prev_allocated : SURFELS_MISSED;
}

bool can_spawn_another_surfel() {
    return atomicMax(unordered_surfels, 0) < MAX_SURFELS_PER_FRAME;
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
    // flag it as currently locked
    atomicOr(surfels[surfel_id].flags, SURFEL_FLAG_LOCKED);

    surfels[surfel_id].instance_id        = instance_id;
    surfels[surfel_id].position_x         = position.x;
    surfels[surfel_id].position_y         = position.y;
    surfels[surfel_id].position_z         = position.z;
    surfels[surfel_id].radius             = radius;
    surfels[surfel_id].normal_x           = normal.x;
    surfels[surfel_id].normal_y           = normal.y;
    surfels[surfel_id].normal_z           = normal.z;
    surfels[surfel_id].irradiance_r       = irradiance.r;
    surfels[surfel_id].irradiance_g       = irradiance.g;
    surfels[surfel_id].irradiance_b       = irradiance.b;
    surfels[surfel_id].contributions      = 1u;
    surfels[surfel_id].morton             = morton_code;

    // set flags to 0 except the lock bit
    atomicAnd(surfels[surfel_id].flags, SURFEL_FLAG_LOCKED);
    
    // set all flags as requested except the lock bit
    atomicOr(surfels[surfel_id].flags, flags & ~SURFEL_FLAG_LOCKED);

    // WARNING: exiting from this function, the surfel is still locked
}

uint binary_search_morton(uint start, uint size, uint key, bool excess) {
    // Non ho voglia di implementarlo bene.

    const uint max_el = start + size;
    uint result = 0;
    for (uint i = start; i < max_el; i++) {
        if ((excess) && (surfels[i].morton > key)) {
            return i;
        } else if ((!excess) && (surfels[i].morton >= key)) {
            return result;
        }
        
        result = i;
    }

    return result;
}

// This is the fast version to search surfels: take advantage of the fact that surfels are
// partially ordered by morton code, and only newer surfels are unordered.
uint linear_search_ordered_surfel_for_allocation(
    in const vec3 eye_position,
    in const vec2 clip_planes,
    vec3 point,
    float radius
) {
    bool too_close = false;

    // Cut (veins) here
    // ==========================================================================
    uint checked_surfels = count_ordered_surfels();
    const uint first_ordered_surfel_id = 0;
    const uint last_ordered_surfel_id = first_ordered_surfel_id + checked_surfels;
    for (uint i = first_ordered_surfel_id; i < last_ordered_surfel_id; i++) {
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
    // ==========================================================================

    // the new surfel can only be allocated if it is distant at least radius
    // from the edge of another surfel. Se we have to search all surfels
    // between begin_colliding_surfel_id and end_colliding_surfel_id and
    // everything in the middle.
    //
    // To do this we calculate morton code of the set of points (called SP) that are
    // "edge-most" of the sphere surfel.origin -> MAX_SURFEL_RADIUS + radius
    // and the search range is min(SP) .. max(SP).
    //
    // This hopefully limits the number of surfels we have to check
    // enough for the algorithm to be very fast.

    const float search_radius = MAX_SURFEL_RADIUS + radius + 0.01; // add a bit of epsilon to avoid precision issues
    vec3 directions[] = {
        vec3(1.000000, 0.000000, 0.000000),
        vec3(-1.000000, 0.000000, 0.000000),
        vec3(0.000000, 1.000000, 0.000000),
        vec3(0.000000, -1.000000, 0.000000),
        vec3(0.000000, 0.000000, 1.000000),
        vec3(0.000000, 0.000000, -1.000000),
        vec3(0.707107, 0.707107, 0.000000),
        vec3(0.707107, -0.707107, 0.000000),
        vec3(-0.707107, 0.707107, 0.000000),
        vec3(-0.707107, -0.707107, 0.000000),
        vec3(0.707107, 0.000000, 0.707107),
        vec3(0.707107, 0.000000, -0.707107),
        vec3(-0.707107, 0.000000, 0.707107),
        vec3(-0.707107, 0.000000, -0.707107),
        vec3(0.000000, 0.707107, 0.707107),
        vec3(0.000000, 0.707107, -0.707107),
        vec3(0.000000, -0.707107, 0.707107),
        vec3(0.000000, -0.707107, -0.707107),
        vec3(0.577350, 0.577350, 0.577350),
        vec3(0.577350, 0.577350, -0.577350),
        vec3(0.577350, -0.577350, 0.577350),
        vec3(0.577350, -0.577350, -0.577350),
        vec3(-0.577350, 0.577350, 0.577350),
        vec3(-0.577350, 0.577350, -0.577350),
        vec3(-0.577350, -0.577350, 0.577350),
        vec3(-0.577350, -0.577350, -0.577350),
    };

    const uint last_ordered_id = count_ordered_surfels();

    uint min_morton = 0xFFFFFFFFu;
    uint max_morton = 0u;
    for (uint i = 0; i < 26; i++) {
        const vec3 edge_point = point + (search_radius * directions[i]);
        const uint morton = morton3D(eye_position, edge_point, clip_planes);
        if (morton == MORTON_OUT_OF_SCALE) {
            min_morton = 0;
            max_morton = last_ordered_id;
            break;
        }

        min_morton = min(min_morton, morton);
        max_morton = max(max_morton, morton);
    }

    const uint begin_colliding_surfel_id = binary_search_morton(0, last_ordered_id, min_morton, false);
    const uint end_colliding_surfel_id = binary_search_morton(0, last_ordered_id, max_morton, true);
    for (uint i = begin_colliding_surfel_id; i <= end_colliding_surfel_id; i++) {
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

// This is the fast version to search surfels: take advantage of the fact that surfels are
// partially ordered by morton code, and only newer surfels are unordered.
uint linear_search_unordered_surfel_for_allocation(
    inout uint checked_surfels,
    vec3 point,
    float radius
) {
    bool too_close = false;

    checked_surfels = count_unordered_surfels();
    const uint first_unordered_surfel_id = total_surfels / 2;
    const uint last_unordered_surfel_id = first_unordered_surfel_id + checked_surfels;
    for (uint i = first_unordered_surfel_id; i < last_unordered_surfel_id; i++) {
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

/*
uint linear_search_surfel(uint last_surfel_id, vec3 point, uint instance_id) {
    for (uint i = 0; i < last_surfel_id; i++) {
        if ((surfels[i].instance_id == instance_id) && (is_point_in_surfel(i, point))) {
            return i;
        }
    }

    return SURFELS_MISSED;
}
*/

#define REGISTER_SURFEL_OK 0
#define REGISTER_SURFEL_FRAME_LIMIT 1
#define REGISTER_SURFEL_FULL 2
#define REGISTER_SURFEL_DENSITY 3
#define REGISTER_SURFEL_OUT_OF_RANGE 4
#define REGISTER_SURFEL_BUSY 5;
#define REGISTER_SURFEL_BELOW_HORIZON 6
#define REGISTER_SURFEL_DISABLED 7
#define REGISTER_SURFEL_IGNORED 8

uint register_surfel(
    uint last_ordered_id,
    in const vec3 eye_position,
    in const vec2 clip_planes,
    uint instance_id,
    bool primary,
    vec3 position,
    float radius,
    vec3 normal,
    vec3 irradiance
) {
#if ENABLE_SURFELS
    uint flags = primary ? SURFEL_FLAG_PRIMARY : 0u;

    uint morton = morton3D(eye_position, position, clip_planes);
    if (morton == MORTON_OUT_OF_SCALE) {
        return REGISTER_SURFEL_OUT_OF_RANGE;
    }

    bool done = false;
    uint checked_surfels = 0;

    // first: do the fast search in the set of ordered surfels set.
    // the ordered set won't change during this shader invocation,
    // so it's safe to do this work only once.
    const uint ordered_surfel_id_search_res = linear_search_ordered_surfel_for_allocation(
        eye_position,
        clip_planes,
        position,
        radius
    );

    if ((ordered_surfel_id_search_res != SURFELS_MISSED) && (ordered_surfel_id_search_res != SURFELS_TOO_CLOSE)) {
        //debugPrintfEXT("|REUSED-ORDERED(%u)", ordered_surfel_id_search_res);
        const uint surfel_id = ordered_surfel_id_search_res;

        // try locking the surfel
        if (!lock_surfel(surfel_id)) {
            // surfel is locked, avoid wasting time and just return busy
            return REGISTER_SURFEL_BUSY;
        }

        // surfel is locked: update it
        const vec3 current_normal = vec3(surfels[surfel_id].normal_x, surfels[surfel_id].normal_y, surfels[surfel_id].normal_z);
        const vec3 current_irradiance = vec3(surfels[surfel_id].irradiance_r, surfels[surfel_id].irradiance_g, surfels[surfel_id].irradiance_b);
        const vec3 new_normal = normalize(current_normal + normal);

        if (dot(current_normal, normal) < 0.0) {
            // surfel is below the horizon: ignore it
            unlock_surfel(surfel_id);
            memoryBarrierBuffer();
            return REGISTER_SURFEL_BELOW_HORIZON;
        }

        surfels[surfel_id].normal_x = new_normal.x;
        surfels[surfel_id].normal_y = new_normal.y;
        surfels[surfel_id].normal_z = new_normal.z;

        surfels[surfel_id].irradiance_r += irradiance.r;
        surfels[surfel_id].irradiance_g += irradiance.g;
        surfels[surfel_id].irradiance_b += irradiance.b;

        surfels[surfel_id].contributions += 1u;

        //const vec3 surfel_diffuse = vec3(surfels[surfel_search_res].irradiance_r, surfels[surfel_search_res].irradiance_g, surfels[surfel_search_res].irradiance_b) / float(surfels[surfel_search_res].irradiance_samples);
        //lights_contribution += contribution(surfel_diffuse, length(light_intensity), diffuse_contribution);

        // surfel is locked: unlock it for other instances
        unlock_surfel(surfel_id);
        memoryBarrierBuffer();

        return REGISTER_SURFEL_OK;
    }

#if FORCE_ALLOCATION
    do {
#endif // FORCE_ALLOCATION
        // try to reuse an existing surfel from the unordered set
        // since the unordered set can change during this shader invocation,
        // I have to either:
        //   - repeat the search until I have searched all surfels that are available at the moment of allocation
        //   - if the allocation fails, because I have missed surfels allocated by parallel shader invocations,
        //     repeat the search with the new number of surfels
        const uint surfel_search_res = linear_search_unordered_surfel_for_allocation(
            checked_surfels,
            position,
            radius
        );
        if ((surfel_search_res != SURFELS_MISSED) && (surfel_search_res != SURFELS_TOO_CLOSE)) {
            //debugPrintfEXT("|REUSED(%u/%u)", surfel_search_res, checked_surfels);
            const uint surfel_id = surfel_search_res;

            // try locking the surfel
            while (!lock_surfel(surfel_id)) {
                // surfel is locked, avoid wasting time and just return busy
                return REGISTER_SURFEL_BUSY;
            }

            // surfel is locked: update it
            const vec3 current_normal = vec3(surfels[surfel_id].normal_x, surfels[surfel_id].normal_y, surfels[surfel_id].normal_z);
            const vec3 current_irradiance = vec3(surfels[surfel_id].irradiance_r, surfels[surfel_id].irradiance_g, surfels[surfel_id].irradiance_b);
            const vec3 new_normal = normalize(current_normal + normal);

            if (dot(current_normal, normal) < 0.0) {
                // surfel is below the horizon: ignore it
                unlock_surfel(surfel_id);
                memoryBarrierBuffer();
                return REGISTER_SURFEL_BELOW_HORIZON;
            }

            surfels[surfel_id].normal_x = new_normal.x;
            surfels[surfel_id].normal_y = new_normal.y;
            surfels[surfel_id].normal_z = new_normal.z;

            surfels[surfel_id].irradiance_r += irradiance.r;
            surfels[surfel_id].irradiance_g += irradiance.g;
            surfels[surfel_id].irradiance_b += irradiance.b;

            surfels[surfel_id].contributions += 1u;

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
                // here we continue the loop to search again
                // since surfels were allocated by other shader invocations meanwhile
                //debugPrintfEXT("|MISSED(%u)", checked_surfels);

                // or, if not forcing allocation, just return REGISTER_SURFEL_IGNORED
            } else {
                init_surfel(surfel_id, flags, instance_id, position, radius, normal, irradiance, morton);
                unlock_surfel(surfel_id);
                memoryBarrierBuffer();
                return REGISTER_SURFEL_OK;

                //debugPrintfEXT("\nCreated surfel %u at position vec3(%f, %f, %f)", surfel_id, surfels[surfel_id].position_x, surfels[surfel_id].position_y, surfels[surfel_id].position_z);
            }
        }
#if FORCE_ALLOCATION
    } while (1 == 1);
#endif // FORCE_ALLOCATION

#else
    return REGISTER_SURFEL_DISABLED;
#endif

    return REGISTER_SURFEL_IGNORED;
}

bool surfel_is_primary(uint surfel_id) {
    return (surfels[surfel_id].flags & SURFEL_FLAG_PRIMARY) != 0u;
}

#endif // _SURFEL_
