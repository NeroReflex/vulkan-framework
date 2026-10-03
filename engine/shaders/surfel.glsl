#ifndef _SURFEL_
#define _SURFEL_

#extension GL_KHR_memory_scope_semantics : require

#include "config.glsl"
#include "random.glsl"
#include "math.glsl"
#include "aabb.glsl"
#include "compress.glsl"
#ifndef SURFEL_NO_SCRATCH
#include "surfel_reorder/build_layout.glsl"
#endif

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
#define SURFELS_BUSY        0xFFFFFFFCu

#define SURFEL_FLAG_LOCKED      (0x01u << 0u)
#define SURFEL_FLAG_PRIMARY     (0x01u << 1u)
#define SURFEL_FLAG_READY       (0x01u << 2u)
#define SURFEL_FLAG_DEAD        (0x01u << 3u)

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

    // Center in the instance's local space. position_* is the world center
    // written by surfel_world.comp. bone == 0xFFFFFFFF means a rigid instance.
    float bind_x;
    float bind_y;
    float bind_z;
    uint bone;
};

// One cache line. Both child bounds sit in the parent so a walk tests
// them without loading the child. 16 words; keep BVH_NODE_SIZE in sync.
struct BVHNode {
    float lmin_x;
    float lmin_y;
    float lmin_z;
    float lmax_x;
    float lmax_y;
    float lmax_z;

    float rmin_x;
    float rmin_y;
    float rmin_z;
    float rmax_x;
    float rmax_y;
    float rmax_z;

    uint left;
    uint right;
    uint parent;
    uint flags;
};

layout (set = SURFELS_DESCRIPTOR_SET, binding = 0, std430) coherent buffer surfel_stats {
    // total number of surfels that can be allocated (max, immutable)
    int total_surfels;

    // Reserved slots in the top half; payload is readable only after READY is acquired.
    int unordered_surfels;

    // Live committed surfels. Indices are stable; the region may contain holes.
    int live_count;

    // Exclusive end of the committed region (live or dead).
    int high_water;

    uint global_reserve_counter;

    uint discovered_surfels;

    // Dead slots still in [0, high_water) after the last commit.
    uint remaining_holes;
    // Set when surfels die or commit; cleared after a full index rebuild.
    uint index_topology_dirty;
};

#ifdef SURFEL_IS_READONLY
readonly
#endif
layout (set = SURFELS_DESCRIPTOR_SET, binding = 1, std430) coherent buffer surfel_buffer_data {
    Surfel surfels[];
};

#ifdef BVH_IS_READONLY
readonly
#endif
layout (set = SURFELS_DESCRIPTOR_SET, binding = 2, std430) buffer surfel_bvh {
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

#ifndef SURFEL_NO_SCRATCH
#ifdef SURFEL_IS_READONLY
readonly
#endif
layout (set = SURFELS_DESCRIPTOR_SET, binding = 7, std430) buffer surfel_build_scratch {
    uint build_words[];
};
#endif

#define NODE_CAPACITY 4096u
layout (set = SURFELS_DESCRIPTOR_SET, binding = 8, std430) readonly buffer NodeWorld {
    mat4 world[NODE_CAPACITY];
} node_world;

#define NODE_IS_LEAF_FLAG 0x80000000u

#ifndef SURFEL_NO_SCRATCH
uint lbvh_key_count() {
    return uint(max(live_count, 0));
}

uint lbvh_root_index() {
    return 0u;
}

uint lbvh_leaf_surfel(uint child) {
    const uint key = child & ~NODE_IS_LEAF_FLAG;
    const uint n = lbvh_key_count();
    if (key >= n) {
        return 0xFFFFFFFFu;
    }
    return build_words[OFF_KEYS_A + key * KEY_STRIDE + 1u];
}

bool lbvh_empty() {
    return lbvh_key_count() == 0u;
}
#endif

// =================== READ SURFEL HELPERS ========================
uint surfel_flags_acquire(uint surfel_id) {
    return atomicLoad(surfels[surfel_id].flags, gl_ScopeDevice,
        gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
}

bool surfel_is_primary(uint surfel_id) {
    return (surfel_flags_acquire(surfel_id) & SURFEL_FLAG_PRIMARY) != 0u;
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
    // live_count is published by the compute build before ray tracing.
    return uint(live_count);
}

uint count_unordered_surfels() {
    // This is a reservation count, not a publication fence for the slot payloads.
    return uint(atomicLoad(unordered_surfels, gl_ScopeDevice,
        gl_StorageSemanticsBuffer, gl_SemanticsRelaxed));
}

// Given the number of UNORDERED surfels already checked (to see if it would have been fitted into any of them),
// allocate a new surfel and return its index. If no surfel is available, return MAX_U32.
uint allocate_surfel(uint checked_surfels) {
    const int scanned = int(checked_surfels);

    // Staging lives in the upper half. Commit fills leftover holes first,
    // then appends up to the committed-half cap.
    const uint committed_cap = uint(total_surfels) / 2u;
    const uint append_room = committed_cap > uint(high_water)
        ? committed_cap - uint(high_water)
        : 0u;
    if (checked_surfels >= remaining_holes + append_room) {
        return SURFELS_FULL;
    }

    uint prev_allocated = atomicCompSwap(unordered_surfels, scanned, scanned + 1,
        gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed,
        gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);

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
    const uint flags = atomicLoad(surfels[surfel_id].flags, gl_ScopeDevice,
        gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
    if ((flags & (SURFEL_FLAG_READY | SURFEL_FLAG_LOCKED)) != SURFEL_FLAG_READY) {
        return false;
    }
    // A single attempt: waiting for another invocation can deadlock a GPU subgroup.
    return atomicCompSwap(surfels[surfel_id].flags, flags, flags | SURFEL_FLAG_LOCKED,
        gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire,
        gl_StorageSemanticsBuffer, gl_SemanticsRelaxed) == flags;
}

void unlock_surfel(uint surfel_id) {
    // Publish payload writes before making the lock available, not after unlocking.
    atomicAnd(surfels[surfel_id].flags, ~SURFEL_FLAG_LOCKED,
        gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
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
    // Reservation gives this invocation exclusive ownership of an upper-half slot.
    atomicStore(surfels[surfel_id].flags, SURFEL_FLAG_LOCKED,
        gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);

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
    surfels[surfel_id].frame_contributions = 0u;
    surfels[surfel_id].latest_contribution = 0u;

    vec3 bind_position = position;
    if (instance_id < NODE_CAPACITY) {
        bind_position = (inverse(node_world.world[instance_id]) * vec4(position, 1.0)).xyz;
    }
    surfels[surfel_id].bind_x = bind_position.x;
    surfels[surfel_id].bind_y = bind_position.y;
    surfels[surfel_id].bind_z = bind_position.z;
    surfels[surfel_id].bone = 0xFFFFFFFFu;

    // Geometry is immutable until the next reorder dispatch. Publish it even if
    // the allocator keeps ownership of the mutable lighting fields.
    const uint published_flags = (flags & ~(SURFEL_FLAG_LOCKED | SURFEL_FLAG_READY))
        | SURFEL_FLAG_READY | (allocate_locked ? SURFEL_FLAG_LOCKED : 0u);
    atomicStore(surfels[surfel_id].flags, published_flags,
        gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
}
#endif // SURFEL_IS_READONLY

bool box_covers_point(in const AABB box, in const vec3 point) {
    return box.vMin.x <= box.vMax.x
        && point.x >= box.vMin.x && point.x <= box.vMax.x
        && point.y >= box.vMin.y && point.y <= box.vMax.y
        && point.z >= box.vMin.z && point.z <= box.vMax.z;
}

AABB left_bounds(uint node) {
    return compatAABB(
        vec3(tree[node].lmin_x, tree[node].lmin_y, tree[node].lmin_z),
        vec3(tree[node].lmax_x, tree[node].lmax_y, tree[node].lmax_z)
    );
}

AABB right_bounds(uint node) {
    return compatAABB(
        vec3(tree[node].rmin_x, tree[node].rmin_y, tree[node].rmin_z),
        vec3(tree[node].rmax_x, tree[node].rmax_y, tree[node].rmax_z)
    );
}

#define LBVH_WALK_LIMIT 64
#define SURFEL_NEIGHBOR_CAP 8

#ifndef SURFEL_NO_SCRATCH
uint hash_bucket(ivec3 cell) {
    uint h = uint(cell.x) * 73856093u ^ uint(cell.y) * 19349663u ^ uint(cell.z) * 83492791u;
    return h & (HASH_BUCKETS - 1u);
}

float hash_cell_size(uint level) {
    return 32.0 * float(1u << level);
}

// Linked list of surfels whose center falls in this cell. Chains are short
// because each surfel is inserted once, into a bucket of its own radius level.
uint hash_chain_head(uint level, ivec3 cell) {
    return build_words[OFF_HASH_HEAD + level * HASH_BUCKETS + hash_bucket(cell)];
}
#endif

void push_bvh(inout uint stack[MAX_BVH_STACK_DEPTH], inout int stack_depth, uint node) {
    if (stack_depth < MAX_BVH_STACK_DEPTH) {
        stack[stack_depth++] = node;
    }
}

uint bvh_search(in const vec3 point) {
#ifndef SURFEL_NO_SCRATCH
    if (lbvh_empty()) {
        return 0xFFFFFFFFu;
    }
    for (uint level = 0u; level < HASH_LEVELS; ++level) {
        const float cs = hash_cell_size(level);
        const ivec3 base = ivec3(floor(point / cs));
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    uint slot = hash_chain_head(level, base + ivec3(dx, dy, dz));
                    for (uint hop = 0u; hop < 32u && slot != HASH_EMPTY && slot < BUILD_HALF; ++hop) {
                        const uint surfel_id = build_words[OFF_HASH_IDS + slot];
                        if (surfel_id != HASH_EMPTY
                            && (surfels[surfel_id].flags & SURFEL_FLAG_DEAD) == 0u
                            && point_inside_surfel(surfel_id, point)) {
                            return surfel_id;
                        }
                        slot = build_words[OFF_HASH_NEXT + slot];
                    }
                }
            }
        }
    }
#endif
    return 0xFFFFFFFFu;
}

uint find_closest_surfel(in const vec3 point) {
#ifndef SURFEL_NO_SCRATCH
    if (lbvh_empty()) {
        return 0xFFFFFFFFu;
    }
    float min_dist = 1e30;
    uint closest_id = 0xFFFFFFFFu;
    for (uint level = 0u; level < HASH_LEVELS; ++level) {
        const float cs = hash_cell_size(level);
        const ivec3 base = ivec3(floor(point / cs));
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    uint slot = hash_chain_head(level, base + ivec3(dx, dy, dz));
                    for (uint hop = 0u; hop < 32u && slot != HASH_EMPTY && slot < BUILD_HALF; ++hop) {
                        const uint surfel_id = build_words[OFF_HASH_IDS + slot];
                        if (surfel_id != HASH_EMPTY && (surfels[surfel_id].flags & SURFEL_FLAG_DEAD) == 0u) {
                            const float d = distance(point, surfelPosition(surfel_id));
                            if (d < min_dist) {
                                min_dist = d;
                                closest_id = surfel_id;
                            }
                        }
                        slot = build_words[OFF_HASH_NEXT + slot];
                    }
                }
            }
        }
    }
    return closest_id;
#else
    return 0xFFFFFFFFu;
#endif
}

uint gather_nearby_surfels(in const vec3 point, in const float radius, inout uint neighbor_ids[SURFEL_NEIGHBOR_CAP]) {
#ifndef SURFEL_NO_SCRATCH
    if (lbvh_empty()) {
        return 0u;
    }
    uint count = 0u;
    const float radius_sq = radius * radius;
    for (uint level = 0u; level < HASH_LEVELS && count < SURFEL_NEIGHBOR_CAP; ++level) {
        const float cs = hash_cell_size(level);
        const ivec3 base = ivec3(floor(point / cs));
        for (int dz = -1; dz <= 1 && count < SURFEL_NEIGHBOR_CAP; ++dz) {
            for (int dy = -1; dy <= 1 && count < SURFEL_NEIGHBOR_CAP; ++dy) {
                for (int dx = -1; dx <= 1 && count < SURFEL_NEIGHBOR_CAP; ++dx) {
                    uint slot = hash_chain_head(level, base + ivec3(dx, dy, dz));
                    for (uint hop = 0u; hop < 32u && slot != HASH_EMPTY && slot < BUILD_HALF && count < SURFEL_NEIGHBOR_CAP; ++hop) {
                        const uint surfel_id = build_words[OFF_HASH_IDS + slot];
                        if (surfel_id != HASH_EMPTY && (surfels[surfel_id].flags & SURFEL_FLAG_DEAD) == 0u) {
                            const vec3 delta = surfelPosition(surfel_id) - point;
                            if (dot(delta, delta) <= radius_sq) {
                                neighbor_ids[count++] = surfel_id;
                            }
                        }
                        slot = build_words[OFF_HASH_NEXT + slot];
                    }
                }
            }
        }
    }
    return count;
#else
    return 0u;
#endif
}

bool is_too_close(vec3 point, float radius, uint surfel_id) {
    const vec3 surfel_center = vec3(surfels[surfel_id].position_x, surfels[surfel_id].position_y, surfels[surfel_id].position_z);
    const float surfel_radius = surfels[surfel_id].radius;
    return distance(point, surfel_center) < (radius + surfel_radius);
}

bool sphere_hits_aabb(in const AABB box, in const vec3 point, in const float radius) {
    const vec3 closest = clamp(point, box.vMin, box.vMax);
    const vec3 delta = point - closest;
    return dot(delta, delta) <= radius * radius;
}

bool surfel_sphere_overlaps(uint surfel_id, in const vec3 point, in const float radius) {
    if ((surfels[surfel_id].flags & SURFEL_FLAG_DEAD) != 0u) {
        return false;
    }
    const float combined = surfels[surfel_id].radius + radius;
    const vec3 delta = surfelPosition(surfel_id) - point;
    return dot(delta, delta) <= combined * combined;
}

// Committed surfels are found by a short hash-chain probe, not a tree walk.
uint linear_search_ordered_surfel_for_allocation(
    vec3 point,
    uint instance_id,
    float radius
) {
#ifndef SURFEL_NO_SCRATCH
    if (lbvh_empty()) {
        return SURFELS_MISSED;
    }

    bool too_close = false;
    for (uint level = 0u; level < HASH_LEVELS; ++level) {
        const float cs = hash_cell_size(level);
        const ivec3 base = ivec3(floor(point / cs));
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    uint slot = hash_chain_head(level, base + ivec3(dx, dy, dz));
                    for (uint hop = 0u; hop < 32u && slot != HASH_EMPTY && slot < BUILD_HALF; ++hop) {
                        const uint surfel_id = build_words[OFF_HASH_IDS + slot];
                        if (surfel_id != HASH_EMPTY && (surfels[surfel_id].flags & SURFEL_FLAG_DEAD) == 0u) {
                            if (is_point_in_surfel(surfel_id, point) && surfels[surfel_id].instance_id == instance_id) {
                                return surfel_id;
                            } else if (is_too_close(point, radius, surfel_id)) {
                                too_close = true;
                            }
                        }
                        slot = build_words[OFF_HASH_NEXT + slot];
                    }
                }
            }
        }
    }

    return too_close ? SURFELS_TOO_CLOSE : SURFELS_MISSED;
#else
    return SURFELS_MISSED;
#endif
}

// This frame's allocations live in a dense prefix of the upper half.
uint linear_search_unordered_surfel_for_allocation(
    inout uint checked_surfels,
    vec3 point,
    uint instance_id,
    float radius
) {
    bool too_close = false;
    bool pending_initialization = false;

    // Snapshot reservations atomically, then acquire each slot's publication
    // separately. Never read a reserved-but-uninitialized payload or wait on its owner.
    checked_surfels = count_unordered_surfels();
    const uint first_unordered_surfel_id = total_surfels / 2;
    const uint last_unordered_surfel_id = first_unordered_surfel_id + checked_surfels;
    for (uint i = first_unordered_surfel_id; i < last_unordered_surfel_id; i++) {
        if ((surfel_flags_acquire(i) & SURFEL_FLAG_READY) == 0u) {
            pending_initialization = true;
            continue;
        }

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

    // Unknown geometry might overlap the proposed allocation. Defer rather than
    // creating a duplicate or spinning until its initialization finishes.
    if (pending_initialization) {
        return SURFELS_BUSY;
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
// via unlock_surfel(), which release-publishes the changes to other shader invocations.
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
    // Far-plane distance only. The old eye-centered AABB slid with the camera
    // and killed stable world surfels every frame while wandering.
    return distance(eye_position, surfel_center) > abs(clip_space.y);
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

// Find an existing committed surfel at a surface point without allocating.
uint find_committed_surfel_at(
    in const vec3 eye_position,
    in const vec2 clip_planes,
    in const uint instance_id,
    in const vec3 position
) {
#if ENABLE_SURFELS
    if (is_out_of_range(eye_position, position, clip_planes)) {
        return 0xFFFFFFFFu;
    }
    const float radius = radius_from_camera_distance(eye_position, clip_planes, position);
    uint id = linear_search_ordered_surfel_for_allocation(position, instance_id, radius);
    if (id != SURFELS_MISSED && id != SURFELS_TOO_CLOSE) {
        return id;
    }
    if (id == SURFELS_TOO_CLOSE) {
        return 0xFFFFFFFFu;
    }
    uint checked = 0u;
    id = linear_search_unordered_surfel_for_allocation(checked, position, instance_id, radius);
    if (id != SURFELS_MISSED && id != SURFELS_TOO_CLOSE && id != SURFELS_BUSY) {
        return id;
    }
#endif
    return 0xFFFFFFFFu;
}

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
// eye_position is the position of the observer: used to calculate morton code and radius size
// clip_planes is the near/far clip planes of the observer: used to calculate morton and discard points too far away
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
    // Retrying under contention is optional, but must never wait indefinitely on peers.
    for (uint attempt = 0u; attempt < 8u; ++attempt) {
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
        if (surfel_search_res == SURFELS_BUSY) {
            return REGISTER_SURFEL_IGNORED;
        } else if ((surfel_search_res != SURFELS_MISSED) && (surfel_search_res != SURFELS_TOO_CLOSE)) {
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
                // Do not debugPrintfEXT here: this is the common per-pixel path
                // once MAX_SURFELS_PER_FRAME is reached. GPU-AV printf on every
                // leftover invocation overflows the printf buffer and TDR's.
                return REGISTER_SURFEL_FULL;
            } else if (surfel_allocation_id == SURFELS_MISSED) {
                // here we continue the loop to search again
                // since surfels were allocated by other shader invocations meanwhile
                //debugPrintfEXT("|MISSED(%u)", checked_surfels);

                // or, if not forcing allocation, just return REGISTER_SURFEL_IGNORED
            } else {
                const uint surfel_id = (total_surfels / 2) + surfel_allocation_id;
                init_surfel(surfel_id, allocate_locked, flags, instance_id, position, radius, normal, diffuse);

                //debugPrintfEXT("clip_planes: vec2(%f, %f), distance: %f, radius: %f\n", clip_planes.x, clip_planes.y, distance(eye_position, position), radius);

                //debugPrintfEXT("\nCreated surfel %u at position vec3(%f, %f, %f)", surfel_id, surfels[surfel_id].position_x, surfels[surfel_id].position_y, surfels[surfel_id].position_z);
                allocated_new = true;
                return surfel_id;
            }
        } else {
            //debugPrintfEXT("\nNICE FUCKUP");
            return REGISTER_SURFEL_VERY_BAD_BUG;
        }
#if FORCE_ALLOCATION
    }
#endif // FORCE_ALLOCATION

#else
    return REGISTER_SURFEL_DISABLED;
#endif

    return REGISTER_SURFEL_IGNORED;
}

#endif // SURFEL_IS_READONLY

#endif // _SURFEL_
