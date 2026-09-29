#ifndef _SURFEL_
#define _SURFEL_

#include "config.glsl"
#include "random.glsl"
#include "math.glsl"
#include "aabb.glsl"
#include "compress.glsl"

#ifndef SURFELS_DESCRIPTOR_SET
#define SURFELS_DESCRIPTOR_SET 5
#endif

#ifndef SURFEL_NO_IMAGES
uniform layout (set = SURFELS_DESCRIPTOR_SET, binding = 4, r32ui) uimage2D surfelOverlappingImage;

uniform layout (set = SURFELS_DESCRIPTOR_SET, binding = 5, rgba32f) image2D outputImage[2];
#endif

#define SURFELS_FULL        0xFFFFFFFFu
#define SURFELS_MISSED      0xFFFFFFFEu
#define SURFELS_TOO_CLOSE   0xFFFFFFFDu

#define SURFEL_FLAG_LOCKED      (0x01u << 0u)
#define SURFEL_FLAG_PRIMARY     (0x01u << 1u)
#define SURFEL_FLAG_DEAD        (0x01u << 2u)

#define SURFEL_GATHER_MAX 64u

#define RADIANCE_THRESHOLD 1.0f

// ensure in std430 the size matches the alignment
// and it is SURFEL_SIZE in rust sources
struct Surfel {
    uint instance_id;

    float position_x;
    float position_y;
    float position_z;
    float radius;

    // the normal is packed using the nvidia method:
    // see compress.glsl
    uint normal;

    float diffuse_r;
    float diffuse_g;
    float diffuse_b;

    float irradiance_r;
    float irradiance_g;
    float irradiance_b;

    float direct_light_r;
    float direct_light_g;
    float direct_light_b;

    // total number of contributions this surfel received sice it was created
    uint contributions;

    // this is the number of contributions this surfel received this frame alone
    uint frame_contributions;

    uint flags;

    // the last time (in frames) this surfel has contributed to the scene
    uint latest_contribution;
};

/**
 * Represents a binary tree node in a linearized bvh tree
 *
 * If you change this remember to also change TLAS_TreeNodeSize
 */
struct BVHNode {
    // these are world positions
    float min_x;
    float min_y;
    float min_z;

    float max_x;
    float max_y;
    float max_z;

    uint parent;
    uint left;
    uint right;

    uint flags;
};

layout (set = SURFELS_DESCRIPTOR_SET, binding = 0, std430) /*coherent*/ buffer surfel_stats {
    // total number of surfels that can be allocated (max, immutable)
    int total_surfels;

    // number of freshly spawned surfels (in the top half of surfels array)
    int unordered_surfels;

    // number of live committed surfels (stable indices, not a dense prefix)
    int live_count;

    // exclusive end of the committed region; slots below this are live or dead
    int high_water;

    uint global_reserve_counter;

    uint discovered_surfels;

    uint padding_2;
    uint padding_3;
};

#ifdef SURFEL_IS_READONLY
readonly
#endif
layout (set = SURFELS_DESCRIPTOR_SET, binding = 1, std430) /*coherent*/ buffer surfel_buffer_data {
    Surfel surfels[];
};

#ifdef BVH_IS_READONLY
readonly
#endif
layout (set = SURFELS_DESCRIPTOR_SET, binding = 2, std430) coherent buffer surfel_bvh {
    BVHNode tree[];
};

#ifndef SURFEL_NO_DISCOVERED
#ifdef DISCOVERED_IS_READONLY
readonly
#endif
layout (set = SURFELS_DESCRIPTOR_SET, binding = 3, std430) /*coherent*/ buffer surfel_discovered {
    uint discovered[];
};
#endif

#define NODE_IS_LEAF_FLAG 0x80000000u

// =================== READ SURFEL HELPERS ========================
bool surfel_is_primary(uint surfel_id) {
    return (surfels[surfel_id].flags & SURFEL_FLAG_PRIMARY) != 0u;
}

vec3 surfelPosition(in Surfel s) {
    return vec3(s.position_x, s.position_y, s.position_z);
}

vec3 surfelNormal(in Surfel s) {
    return decompress_unit_vec(s.normal);
}

vec3 surfelPosition(uint surfel_id) {
    return surfelPosition(surfels[surfel_id]);
}

vec3 surfelNormal(uint surfel_id) {
    return surfelNormal(surfels[surfel_id]);
}

// Calculate the light given from the surfel to the given position, assuming no object in-between:
// basically use a surfel as a point light source.
vec3 projected_irradiance(in Surfel s, in const vec3 position, in const vec3 normal) {
    const vec3 reflected_directional_light = vec3(
        s.direct_light_r,
        s.direct_light_g,
        s.direct_light_b
    );

    const vec3 diffuse = vec3(s.diffuse_r, s.diffuse_g, s.diffuse_b);
    
    vec3 irradiance = vec3(0.0);
    if (s.contributions > 0u) {
        irradiance = vec3(
            s.irradiance_r / float(s.contributions),
            s.irradiance_g / float(s.contributions),
            s.irradiance_b / float(s.contributions)
        );
    }

    const vec3 surfel_pos = surfelPosition(s);
    if (distance(surfel_pos, position) < kEpsilon) {
        return (reflected_directional_light + irradiance) * diffuse;
    }

    const vec3 light_dir = normalize(surfel_pos - position);
    const float intensity = max(dot(normal, light_dir), 0.0);

    return intensity * diffuse * (
        reflected_directional_light + irradiance
    );
}

vec3 projected_irradiance(const uint surfel_id, in const vec3 position, in const vec3 normal) {
    return projected_irradiance(surfels[surfel_id], position, normal);
}

AABB surfelAABB(uint surfel_id) {
    return compatAABB(
        surfelPosition(surfel_id) - surfels[surfel_id].radius,
        surfelPosition(surfel_id) + surfels[surfel_id].radius
    );
}

bool point_inside_surfel(uint surfel_id, vec3 point) {
    return distance(surfelPosition(surfel_id), point) <= surfels[surfel_id].radius;
}
// =================================================================

// =================== WRITE SURFEL HELPERS ========================
#ifndef SURFEL_IS_READONLY

void addDirectionalLightContribution(uint surfel_id, vec3 received_directional_light) {
    surfels[surfel_id].direct_light_r += received_directional_light.r;
    surfels[surfel_id].direct_light_g += received_directional_light.g;
    surfels[surfel_id].direct_light_b += received_directional_light.b;
}

void addIndirectLightContribution(uint surfel_id, vec3 received_light) {
    const uint max_contributions = 16384u;
    if (surfels[surfel_id].contributions > max_contributions) {
        // avoid overflow and at the same time help in dynamic scenes
        surfels[surfel_id].irradiance_r -= surfels[surfel_id].irradiance_r / float(max_contributions);
        surfels[surfel_id].irradiance_g -= surfels[surfel_id].irradiance_g / float(max_contributions);
        surfels[surfel_id].irradiance_b -= surfels[surfel_id].irradiance_b / float(max_contributions);
    }

    surfels[surfel_id].irradiance_r += received_light.r;
    surfels[surfel_id].irradiance_g += received_light.g;
    surfels[surfel_id].irradiance_b += received_light.b;

    surfels[surfel_id].contributions += 1u;
}

#endif // SURFEL_IS_READONLY
// =================================================================

bool is_point_in_surfel(uint surfel_id, const in vec3 point) {
    const vec3 center = surfelPosition(surfel_id);
    const float radius = surfels[surfel_id].radius;

    const vec3 direction = point - center;

    //return length(direction) <= radius;

    // this is supposed to be more efficient
    return dot(direction, direction) <= radius * radius;
}

uint count_discoveder_surfels() {
    // discovered_surfels is never modified in raytrace shader,
    // so avoid the expensive atomic read
    return discovered_surfels;
}

uint count_ordered_surfels() {
    // live_count is published by the compute build, before ray tracing runs,
    // so avoid the expensive atomic read
    return uint(live_count);
}

uint count_unordered_surfels() {
    // Once a new surfel is allocated, unordered_surfels is incremented,
    // and a memoryBufferBarrier() is issued: making this value coherent again:
    // avoid an expensive atomic read.
    return unordered_surfels;

    //return atomicMax(unordered_surfels, 0);
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
    // it's not THAT much important to be exact here, so avoid the expensive atomic read
    // since the allocation will fail if we ran out of space anyway, the only downside
    // is that we might ends up allocating more surfels than the allowed maximum per frame,
    // but not more than what shader(s) can handle 
    //return atomicMax(unordered_surfels, 0) < MAX_SURFELS_PER_FRAME;
    return count_unordered_surfels() < MAX_SURFELS_PER_FRAME;
}

#ifndef SURFEL_IS_READONLY
bool lock_surfel(uint surfel_id) {
    return (atomicOr(surfels[surfel_id].flags, SURFEL_FLAG_LOCKED) & SURFEL_FLAG_LOCKED) == 0u;
}

void unlock_surfel(uint surfel_id) {
    atomicAnd(surfels[surfel_id].flags, ~SURFEL_FLAG_LOCKED);
}

void init_surfel(
    uint surfel_id,
    in const bool allocate_locked,
    uint flags,
    in const uint instance_id,
    in const vec3 position,
    in const float radius,
    in const vec3 normal,
    in const vec3 diffuse
) {
    // flag it as currently locked
    atomicOr(surfels[surfel_id].flags, SURFEL_FLAG_LOCKED);

    surfels[surfel_id].instance_id    = instance_id;
    surfels[surfel_id].position_x     = position.x;
    surfels[surfel_id].position_y     = position.y;
    surfels[surfel_id].position_z     = position.z;
    surfels[surfel_id].radius         = radius;
    surfels[surfel_id].normal         = compress_unit_vec(normal);
    surfels[surfel_id].diffuse_r      = diffuse.r;
    surfels[surfel_id].diffuse_g      = diffuse.g;
    surfels[surfel_id].diffuse_b      = diffuse.b;
    surfels[surfel_id].irradiance_r   = 0;
    surfels[surfel_id].irradiance_g   = 0;
    surfels[surfel_id].irradiance_b   = 0;
    surfels[surfel_id].direct_light_r = 0;
    surfels[surfel_id].direct_light_g = 0;
    surfels[surfel_id].direct_light_b = 0;
    surfels[surfel_id].contributions  = 0u;
    surfels[surfel_id].latest_contribution = 0u;

    // set flags to 0 except the lock bit
    atomicAnd(surfels[surfel_id].flags, SURFEL_FLAG_LOCKED);

    // set all flags as requested (lock bit is special)
    atomicOr(
        surfels[surfel_id].flags,
        flags & ((!allocate_locked) ? ~SURFEL_FLAG_LOCKED : 0xFFFFFFFFu));

    // WARNING: exiting from this function, the surfel is still locked
}
#endif // SURFEL_IS_READONLY

uint bvh_search(in const vec3 point) {
    // BVH is empty: return a failure.
    if (tree[0].left == tree[0].parent) {
        return 0xFFFFFFFFu;
    }

    uint stack[MAX_BVH_STACK_DEPTH];
    int stackDepth = 0;

    uint currentIndex = 0;
    stack[stackDepth++] = currentIndex;

    while (stackDepth > 0) {
        const uint childR = tree[currentIndex].right;
        const uint childL = tree[currentIndex].left;

        const bool leftIsLeaf = (childL & NODE_IS_LEAF_FLAG) != 0;
        const bool rightIsLeaf = (childR & NODE_IS_LEAF_FLAG) != 0;

        const uint rightIdx = (childR & ~(NODE_IS_LEAF_FLAG));
        const uint leftIdx = (childL & ~(NODE_IS_LEAF_FLAG));

        bool traverseL = false;
        bool traverseR = false;

        if (leftIsLeaf) {
            if (point_inside_surfel(leftIdx, point)) {
                return leftIdx;
            }
        } else {
            const AABB leftAABB = compatAABB(
                vec3(tree[leftIdx].min_x, tree[leftIdx].min_y, tree[leftIdx].min_z),
                vec3(tree[leftIdx].max_x, tree[leftIdx].max_y, tree[leftIdx].max_z)
            );

            traverseL = AABBcontains(
                leftAABB,
                point
            );
        }

        if (rightIsLeaf) {
            if (point_inside_surfel(rightIdx, point)) {
                return rightIdx;
            }
        } else {
            const AABB rightAABB = compatAABB(
                vec3(tree[rightIdx].min_x, tree[rightIdx].min_y, tree[rightIdx].min_z),
                vec3(tree[rightIdx].max_x, tree[rightIdx].max_y, tree[rightIdx].max_z)
            );

            traverseR = AABBcontains(
                rightAABB,
                point
            );
        }

        if ((!traverseL) && (!traverseR)) {
            currentIndex = stack[--stackDepth];
        } else {
            currentIndex = (traverseL) ? leftIdx : rightIdx;

            if (traverseL && traverseR) {
                //if (stackDepth == MAX_BVH_STACK_DEPTH) {
                //    debugPrintfEXT("Max search stack depth reached.");
                //}

                stack[stackDepth++] = rightIdx;
            }
        }
    }

    return 0xFFFFFFFFu;
}

// BVH-based closest surfel search
uint find_closest_surfel(in const vec3 point) {
    if (tree[0].left == tree[0].parent) {
        return 0xFFFFFFFFu;
    }

    uint stack[MAX_BVH_STACK_DEPTH];
    int stackDepth = 0;
    uint currentIndex = 0;
    stack[stackDepth++] = currentIndex;

    float min_dist = 1e30;
    uint closest_id = 0xFFFFFFFFu;

    while (stackDepth > 0) {
        currentIndex = stack[--stackDepth];
        const uint childR = tree[currentIndex].right;
        const uint childL = tree[currentIndex].left;

        const bool leftIsLeaf = (childL & NODE_IS_LEAF_FLAG) != 0;
        const bool rightIsLeaf = (childR & NODE_IS_LEAF_FLAG) != 0;

        const uint rightIdx = (childR & ~(NODE_IS_LEAF_FLAG));
        const uint leftIdx = (childL & ~(NODE_IS_LEAF_FLAG));

        // Check left child
        if (leftIsLeaf) {
            float d = distance(point, surfelPosition(leftIdx));
            if (d < min_dist) {
                min_dist = d;
                closest_id = leftIdx;
            }
        } else {
            const AABB leftAABB = compatAABB(
                vec3(tree[leftIdx].min_x, tree[leftIdx].min_y, tree[leftIdx].min_z),
                vec3(tree[leftIdx].max_x, tree[leftIdx].max_y, tree[leftIdx].max_z)
            );
            float aabb_dist = distanceAABBPoint(leftAABB, point);
            if (aabb_dist < min_dist) {
                stack[stackDepth++] = leftIdx;
            }
        }

        // Check right child
        if (rightIsLeaf) {
            float d = distance(point, surfelPosition(rightIdx));
            if (d < min_dist) {
                min_dist = d;
                closest_id = rightIdx;
            }
        } else {
            const AABB rightAABB = compatAABB(
                vec3(tree[rightIdx].min_x, tree[rightIdx].min_y, tree[rightIdx].min_z),
                vec3(tree[rightIdx].max_x, tree[rightIdx].max_y, tree[rightIdx].max_z)
            );
            float aabb_dist = distanceAABBPoint(rightAABB, point);
            if (aabb_dist < min_dist) {
                stack[stackDepth++] = rightIdx;
            }
        }
    }
    return closest_id;
}

bool sphere_hits_aabb(in const AABB box, in const vec3 point, in const float radius) {
    return distanceAABBPoint(box, point) <= radius;
}

bool is_too_close(vec3 point, float radius, uint surfel_id) {
    const vec3 surfel_center = vec3(surfels[surfel_id].position_x, surfels[surfel_id].position_y, surfels[surfel_id].position_z);
    const float surfel_radius = surfels[surfel_id].radius;
    return distance(point, surfel_center) < (radius + surfel_radius);
}

// Visit every committed surfel whose sphere can overlap `point` within `radius`.
// The median-split tree is height ceil(log2(n)); a visit is O(log n + k).
// `on_leaf` is inlined at the two call sites below.
bool surfel_sphere_overlaps(uint surfel_id, in const vec3 point, in const float radius) {
    return distance(point, surfelPosition(surfel_id)) <= (radius + surfels[surfel_id].radius);
}

struct SurfelGather {
    uint count;
    uint ids[SURFEL_GATHER_MAX];
};

// Collect committed surfels whose sphere overlaps the query sphere.
// The upper half (this frame's allocations) is not in the tree.
SurfelGather gather_nearby_surfels(in const vec3 point, in const float radius) {
    SurfelGather result;
    result.count = 0u;

    uint stack[MAX_BVH_STACK_DEPTH];
    int stack_depth = 0;
    if (tree[0].left != tree[0].parent) {
        stack[stack_depth++] = 0u;
    }

    while (stack_depth > 0) {
        const uint node = stack[--stack_depth];
        const uint child_l = tree[node].left;
        const uint child_r = tree[node].right;
        const bool left_is_leaf = (child_l & NODE_IS_LEAF_FLAG) != 0u;
        const bool right_is_leaf = (child_r & NODE_IS_LEAF_FLAG) != 0u;
        const uint left_idx = child_l & ~NODE_IS_LEAF_FLAG;
        const uint right_idx = child_r & ~NODE_IS_LEAF_FLAG;

        if (left_is_leaf) {
            if (surfel_sphere_overlaps(left_idx, point, radius) && result.count < SURFEL_GATHER_MAX) {
                result.ids[result.count++] = left_idx;
            }
        } else {
            const AABB left_box = compatAABB(
                vec3(tree[left_idx].min_x, tree[left_idx].min_y, tree[left_idx].min_z),
                vec3(tree[left_idx].max_x, tree[left_idx].max_y, tree[left_idx].max_z)
            );
            if (sphere_hits_aabb(left_box, point, radius) && stack_depth < MAX_BVH_STACK_DEPTH) {
                stack[stack_depth++] = left_idx;
            }
        }

        // A single committed surfel is stored in both children. Emit it once.
        if (right_is_leaf && left_is_leaf && right_idx == left_idx) {
            continue;
        }

        if (right_is_leaf) {
            if (surfel_sphere_overlaps(right_idx, point, radius) && result.count < SURFEL_GATHER_MAX) {
                result.ids[result.count++] = right_idx;
            }
        } else {
            const AABB right_box = compatAABB(
                vec3(tree[right_idx].min_x, tree[right_idx].min_y, tree[right_idx].min_z),
                vec3(tree[right_idx].max_x, tree[right_idx].max_y, tree[right_idx].max_z)
            );
            if (sphere_hits_aabb(right_box, point, radius) && stack_depth < MAX_BVH_STACK_DEPTH) {
                stack[stack_depth++] = right_idx;
            }
        }
    }

    // This frame's allocations are not in the tree. There are at most 256 of them.
    const uint fresh = min(uint(unordered_surfels), uint(MAX_SURFELS_PER_FRAME));
    const uint upper_base = uint(total_surfels) / 2u;
    for (uint i = 0u; i < fresh && result.count < SURFEL_GATHER_MAX; ++i) {
        const uint id = upper_base + i;
        if (surfel_sphere_overlaps(id, point, radius)) {
            result.ids[result.count++] = id;
        }
    }

    return result;
}

// Committed surfels are found by a radius walk of the median-split tree.
// This frame's new surfels are not in the tree; the caller scans those separately.
uint linear_search_ordered_surfel_for_allocation(
    vec3 point,
    uint instance_id,
    float radius
) {
    if (tree[0].left == tree[0].parent) {
        return SURFELS_MISSED;
    }

    bool too_close = false;
    uint stack[MAX_BVH_STACK_DEPTH];
    int stack_depth = 0;
    stack[stack_depth++] = 0u;

    while (stack_depth > 0) {
        const uint node = stack[--stack_depth];
        const uint child_l = tree[node].left;
        const uint child_r = tree[node].right;
        const bool left_is_leaf = (child_l & NODE_IS_LEAF_FLAG) != 0u;
        const bool right_is_leaf = (child_r & NODE_IS_LEAF_FLAG) != 0u;
        const uint left_idx = child_l & ~NODE_IS_LEAF_FLAG;
        const uint right_idx = child_r & ~NODE_IS_LEAF_FLAG;

        if (left_is_leaf) {
            if ((is_point_in_surfel(left_idx, point)) && (surfels[left_idx].instance_id == instance_id)) {
                return left_idx;
            } else if (is_too_close(point, radius, left_idx)) {
                too_close = true;
            }
        } else {
            const AABB left_box = compatAABB(
                vec3(tree[left_idx].min_x, tree[left_idx].min_y, tree[left_idx].min_z),
                vec3(tree[left_idx].max_x, tree[left_idx].max_y, tree[left_idx].max_z)
            );
            if (sphere_hits_aabb(left_box, point, radius) && stack_depth < MAX_BVH_STACK_DEPTH) {
                stack[stack_depth++] = left_idx;
            }
        }

        if (right_is_leaf && left_is_leaf && right_idx == left_idx) {
            continue;
        }

        if (right_is_leaf) {
            if ((is_point_in_surfel(right_idx, point)) && (surfels[right_idx].instance_id == instance_id)) {
                return right_idx;
            } else if (is_too_close(point, radius, right_idx)) {
                too_close = true;
            }
        } else {
            const AABB right_box = compatAABB(
                vec3(tree[right_idx].min_x, tree[right_idx].min_y, tree[right_idx].min_z),
                vec3(tree[right_idx].max_x, tree[right_idx].max_y, tree[right_idx].max_z)
            );
            if (sphere_hits_aabb(right_box, point, radius) && stack_depth < MAX_BVH_STACK_DEPTH) {
                stack[stack_depth++] = right_idx;
            }
        }
    }

    return too_close ? SURFELS_TOO_CLOSE : SURFELS_MISSED;
}

// This frame's allocations live in a dense prefix of the upper half.
// The counter is published with memoryBarrierBuffer() after each init.
uint linear_search_unordered_surfel_for_allocation(
    inout uint checked_surfels,
    vec3 point,
    uint instance_id,
    float radius
) {
    bool too_close = false;

    // This is very clever (and probably based on UB): the number of unordered surfels
    // is only increase via atomic operations, but since this function reads it in a
    // non-atomic way, it is VERY LIKELY to read a stale value, that will be updated
    // only when a concurrently-allocating-surfel thread has committed the updated
    // counter, as well as the new surfel position via a memoryBufferBarrier(), thus
    // ensuring I won't spawn a new surfel that would be too close to an already allocated one
    // simply because I haven't read that surfel yet.
    checked_surfels = count_unordered_surfels();
    const uint first_unordered_surfel_id = total_surfels / 2;
    const uint last_unordered_surfel_id = first_unordered_surfel_id + checked_surfels;
    for (uint i = first_unordered_surfel_id; i < last_unordered_surfel_id; i++) {
        if ((is_point_in_surfel(i, point)) && (surfels[i].instance_id == instance_id)) {
            return i;
        }

        if (distance(point, surfelPosition(i)) < (radius + surfels[i].radius)) {
            too_close = true;
            // do not break, we want to check all surfels for matches,
            // but we also want to know if we were too close to any of them
            // to avoid allocating new ones
        }
    }

    return too_close ? SURFELS_TOO_CLOSE : SURFELS_MISSED;
}

#ifndef SURFEL_IS_READONLY

#define UPDATE_SURFEL_OK 0
#define UPDATE_SURFEL_BUSY 0xFFFFFFFFu

// Update the surfel with the given id, adding the given irradiance.
//
//
// WARNING: this function MUST be called only after successfully locking the surfel
// via lock_surfel(), and the surfel MUST be unlocked after this function returns
// via unlock_surfel(), MOREOVER the caller is responsible for placing memoryBarrierBuffer()
// after unlocking the surfel, to ensure the changes are visible to other shader invocations.
uint add_diffuse_sample_to_surfel(
    uint surfel_id,
    vec3 normal,
    vec3 irradiance
) {
    const vec3 surfel_center = surfelPosition(surfel_id);
    const vec3 surfel_normal = surfelNormal(surfel_id);
    const vec3 current_irradiance = vec3(surfels[surfel_id].irradiance_r, surfels[surfel_id].irradiance_g, surfels[surfel_id].irradiance_b);

    surfels[surfel_id].irradiance_r += irradiance.r;
    surfels[surfel_id].irradiance_g += irradiance.g;
    surfels[surfel_id].irradiance_b += irradiance.b;

    surfels[surfel_id].contributions += 1u;

    return UPDATE_SURFEL_OK;
}

#endif // SURFEL_IS_READONLY

bool is_out_of_range(in const vec3 eye_position, in const vec3 surfel_center, in const vec2 clip_space) {
    const float range = abs(clip_space.y) - abs(clip_space.x);

    const vec3 min_allowed_position = eye_position - vec3(range);
    const vec3 max_allowed_position = eye_position + vec3(range);

    return any(lessThan(surfel_center, min_allowed_position)) || any(greaterThan(surfel_center, max_allowed_position));
}

float radius_from_camera_distance(
    in const vec3 eye_position,
    in const vec2 clip_planes,
    in const vec3 position
) {
    // this is a linear mapping from distance to radius
    // that maps 0.0 -> MIN_SURFEL_RADIUS and 1.0 -> MAX_SURFEL_RADIUS
    return clamp(
        map(
            distance(eye_position, position),
            abs(clip_planes.x),
            abs(clip_planes.y),
            MIN_SURFEL_RADIUS,
            MAX_SURFEL_RADIUS
        ),
        MIN_SURFEL_RADIUS,
        MAX_SURFEL_RADIUS
    );
}

#define IS_SURFEL_VALID(surfel_id) (surfel_id < total_surfels)

#ifndef SURFEL_IS_READONLY

#define REGISTER_SURFEL_VERY_BAD_BUG 0xFFFFFFF8u
#define REGISTER_SURFEL_FRAME_LIMIT 0xFFFFFFF9u
#define REGISTER_SURFEL_FULL 0xFFFFFFFAu
#define REGISTER_SURFEL_DENSITY 0xFFFFFFFBu
#define REGISTER_SURFEL_OUT_OF_RANGE 0xFFFFFFFCu
#define REGISTER_SURFEL_BELOW_HORIZON 0xFFFFFFFDu
#define REGISTER_SURFEL_DISABLED 0xFFFFFFFEu
#define REGISTER_SURFEL_IGNORED 0xFFFFFFFFu

// Function to register contribution to a surfel, or allocate a new one if needed
// This function searches for a surfel in a compatible position first in the set of ordered
// surfels (where surfels can be updated, but the set itself won't change during this shader invocation),
// and then in the unordered set (where surfels can be updated, but also new surfels can be allocated)
// and if no compatible surfel is found, and no surfel is too close, a new surfel is allocated.
//
// eye_position is the position of the observer: used to calculate radius and discard points too far away
// clip_planes is the near/far clip planes of the observer
// instance_id is the instance id of the object generating the surfel
// position is the position of the surfel to register
// normal is the normal of the surfel to register
// diffuse is the diffuse color of the surfel to register
// irradiance is the irradiance of the surfel to register
// allocate_locked leave new allocations locked
// allocated_new is set to true if a new surfel was allocated
uint find_surfel_or_allocate_new(
    in const vec3 eye_position,
    in const vec2 clip_planes,
    in const uint instance_id,
    in const vec3 position,
    in const vec3 normal,
    in const vec3 diffuse,
    in const bool allocate_locked,
    out bool allocated_new
) {
    allocated_new = false;
    if (is_out_of_range(eye_position, position, clip_planes)) {
        return REGISTER_SURFEL_OUT_OF_RANGE;
    }

#if ENABLE_SURFELS
    uint flags = 0u;

    const float radius = radius_from_camera_distance(eye_position, clip_planes, position);

    bool done = false;
    uint checked_surfels = 0;

    // first: do the fast search in the set of ordered surfels set.
    // the ordered set won't change during this shader invocation,
    // so it's safe to do this work only once.
    const uint ordered_surfel_id_search_res = linear_search_ordered_surfel_for_allocation(
        position,
        instance_id,
        radius
    );

    if ((ordered_surfel_id_search_res != SURFELS_MISSED) && (ordered_surfel_id_search_res != SURFELS_TOO_CLOSE)) {
        return ordered_surfel_id_search_res;
    } else if (ordered_surfel_id_search_res == SURFELS_TOO_CLOSE) {
        // we were too close to an existing surfel: do not allocate a new one
        //debugPrintfEXT("|TOO CLOSE");
        return REGISTER_SURFEL_DENSITY;
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
            instance_id,
            radius
        );
        if ((surfel_search_res != SURFELS_MISSED) && (surfel_search_res != SURFELS_TOO_CLOSE)) {
            return surfel_search_res;
        } else if (surfel_search_res == SURFELS_TOO_CLOSE) {
            // we were too close to an existing surfel: do not allocate a new one
            //debugPrintfEXT("|TOO CLOSE");
            return REGISTER_SURFEL_DENSITY;
        } else if (!can_spawn_another_surfel()) {
            // we cannot allocate more surfels this frame
            // to avoid impacting too much on the frame time
            //debugPrintfEXT("|FRAME_LIMIT(1)");
            return REGISTER_SURFEL_FRAME_LIMIT;
        } else if (surfel_search_res == SURFELS_MISSED) {
            // A matching surfel was not found: try to allocate a new one
            uint surfel_allocation_id = allocate_surfel(checked_surfels);
            if (surfel_allocation_id == SURFELS_FULL) {
                debugPrintfEXT("|FULL(%u)", checked_surfels);
                return REGISTER_SURFEL_FULL;
            } else if (surfel_allocation_id == SURFELS_MISSED) {
                // here we continue the loop to search again
                // since surfels were allocated by other shader invocations meanwhile
                //debugPrintfEXT("|MISSED(%u)", checked_surfels);

                // or, if not forcing allocation, just return REGISTER_SURFEL_IGNORED
            } else {
                const uint surfel_id = (total_surfels / 2) + surfel_allocation_id;
                init_surfel(surfel_id, allocate_locked, flags, instance_id, position, radius, normal, diffuse);
                unlock_surfel(surfel_id);
                memoryBarrierBuffer();

                //debugPrintfEXT("clip_planes: vec2(%f, %f), distance: %f, radius: %f\n", clip_planes.x, clip_planes.y, distance(eye_position, position), radius);

                //debugPrintfEXT("\nCreated surfel %u at position vec3(%f, %f, %f)", surfel_id, surfels[surfel_id].position_x, surfels[surfel_id].position_y, surfels[surfel_id].position_z);
                allocated_new = true;
                return surfel_id;
            }
        } else {
            debugPrintfEXT("\nNICE FUCKUP");
            return REGISTER_SURFEL_VERY_BAD_BUG;
        }
#if FORCE_ALLOCATION
    } while (1 == 1);
#endif // FORCE_ALLOCATION

#else
    return REGISTER_SURFEL_DISABLED;
#endif

    return REGISTER_SURFEL_IGNORED;
}

#endif // SURFEL_IS_READONLY

#endif // _SURFEL_
