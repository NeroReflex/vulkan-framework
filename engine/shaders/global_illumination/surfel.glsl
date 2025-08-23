#ifndef _SURFEL_
#define _SURFEL_

#include "../config.glsl"

#define SURFELS_FULL 0xFFFFFFFFu
#define SURFELS_MISSED 0xFFFFFFFEu

struct SurfelStats {
    uint total_surfels;
    uint free_surfels;
    uint busy_surfels;
};

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

    // 0 free, 1 busy
    uint lock;

    uint padding[2];
};

layout (set = 5, binding = 0, std430) coherent buffer surfel_stats {
    SurfelStats stats;
};

layout (set = 5, binding = 1, std430) coherent buffer surfel_buffer_data { 
    Surfel surfels[];
};

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
    if (checked_surfels >= stats.total_surfels) {
        return SURFELS_FULL;
    }

    uint prev_allocated = atomicCompSwap(stats.busy_surfels, checked_surfels, checked_surfels + 1);

    return prev_allocated == checked_surfels ? prev_allocated : SURFELS_MISSED;
}

uint allocated_surfels() {
    return atomicMax(stats.busy_surfels, 0);
}

bool lock_surfel(uint surfel_id) {
    return atomicCompSwap(surfels[surfel_id].lock, 0u, 1u) == 0;
}

void unlock_surfel(uint surfel_id) {
    atomicAnd(surfels[surfel_id].lock, 0u);
}

void init_surfel(uint surfel_id, uint instance_id, vec3 position, float radius, vec3 normal, vec3 irradiance) {
    atomicOr(surfels[surfel_id].lock, 1u);
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
    unlock_surfel(surfel_id);
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

#endif // _SURFEL_
