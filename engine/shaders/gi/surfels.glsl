#ifndef _GI_SURFELS_
#define _GI_SURFELS_

// Requires status.glsl (gi_data) and SURFEL_SET.
#include "common.glsl"

#ifndef SURFEL_SET
#error "SURFEL_SET not defined"
#endif

#define SURFEL_FLAG_ALIVE (1u << 0u)
#define SURFEL_FLAG_DOOMED (1u << 1u)
#define BONE_RIGID 0xFFFFFFFFu

// rust: SURFEL_STRIDE = 128 bytes
struct Surfel {
    vec3 position;
    float radius;
    vec3 bind_position;
    uint instance_id;
    uint normal_bind;
    uint bone;
    uint instance_generation;
    // bumped on every allocation of this slot; reservoirs store it to detect reuse
    uint generation;
    vec3 albedo;
    // partition-of-unity area estimate (EMA)
    float area;
    // incident direct irradiance (EMA) and its sample count
    vec3 e_dir;
    float e_dir_n;
    // incident indirect irradiance (EMA) and its sample count
    vec3 e_ind;
    float e_ind_n;
    vec3 normal;
    uint flags;
    uint last_seen;
    float area_n;
    uint spawn_frame;
    // first frame whose publish pass made this surfel a light (INVALID_ID = not yet)
    uint lit_frame;
};

// rust: ST_* in gi_math.rs
#define ST_ALIVE_IN 0
#define ST_ALIVE_OUT 1
#define ST_SURVIVORS 2
#define ST_FREE 3
#define ST_SPAWN_SCREEN 4
#define ST_SPAWN_RAY 5
#define ST_EVICTED 6
#define ST_LEVEL_MASK 7
#define ST_GRID_ITEMS 8
// sum of the quantized fluxes (uint): the last entry of flux_cdf
#define ST_TOTAL_FLUX 9
#define ST_LIGHT_COUNT 10
#define ST_FREE_AT_BEGIN 11
#define ST_SPAWN_FAILED 12
#define ST_DOOMED 13
// largest light flux of this frame / of the previous frame (float bits); the
// previous one scales the flux quantization of the CDF
#define ST_MAX_FLUX 14
#define ST_PREV_MAX_FLUX 15

layout(set = SURFEL_SET, binding = 0, std430) coherent buffer SurfelState { int state[]; };
layout(set = SURFEL_SET, binding = 1, std430) buffer SurfelPool { Surfel surfels[]; };
layout(set = SURFEL_SET, binding = 2, std430) buffer SurfelFree { uint free_ids[]; };
// two lists of MAX_SURFELS ids; the frame parity selects which one is written
layout(set = SURFEL_SET, binding = 3, std430) buffer SurfelAlive { uint alive_ids[]; };
// compact per-slot data for queries: (center, radius) and the octahedral world normal
layout(set = SURFEL_SET, binding = 4, std430) buffer SurfelGeo { vec4 geo[]; };
layout(set = SURFEL_SET, binding = 5, std430) buffer SurfelNormal { uint nrm[]; };
// (L * A, A): emitted radiance times area, and area. Written by the publish pass.
layout(set = SURFEL_SET, binding = 6, std430) buffer SurfelLight { vec4 surfel_light[]; };

layout(set = SURFEL_SET, binding = 7, std430) buffer GridCount { uint grid_count[]; };
layout(set = SURFEL_SET, binding = 8, std430) buffer GridStart { uint grid_start[]; };
layout(set = SURFEL_SET, binding = 9, std430) buffer GridCursor { uint grid_cursor[]; };
layout(set = SURFEL_SET, binding = 10, std430) buffer GridItems { uint grid_items[]; };

// Inclusive prefix sum of the quantized fluxes in [1, FLUX_QUANT_MAX]: integer,
// hence exactly monotonic, and the sum over MAX_SURFELS entries fits 32 bits.
layout(set = SURFEL_SET, binding = 11, std430) buffer FluxCdf { uint flux_cdf[]; };
layout(set = SURFEL_SET, binding = 12, std430) buffer ScanBlocks { uint scan_blocks[]; };
// A presampled light carries what initial candidates need, so a work group
// reading one tile stays in cache. rust: TILE_ENTRY_STRIDE = 32 bytes
struct TileEntry {
    // light center and radius
    vec4 sphere;
    // id (INVALID_ID = empty), packed normal, flux bits, pmf bits
    uvec4 info;
};
layout(set = SURFEL_SET, binding = 13, std430) buffer LightTiles { TileEntry light_tiles[]; };

#define NODE_CAPACITY 4096u
#define MAX_PALETTE_BONES 4096u
layout(set = SURFEL_SET, binding = 15, std430) readonly buffer NodeWorld { mat4 world[NODE_CAPACITY]; } node_world;
layout(set = SURFEL_SET, binding = 16, std430) readonly buffer BonePalette { mat4 bone[MAX_PALETTE_BONES]; } bone_palette;
layout(set = SURFEL_SET, binding = 17, std430) readonly buffer InstanceGeneration { uint generation[NODE_CAPACITY]; } instance_generation;
layout(set = SURFEL_SET, binding = 18, std430) readonly buffer InstanceBoneBase { uint bone_base[NODE_CAPACITY]; } instance_bone_base;

// Compact per-surfel light state read by ReSTIR, written by publish.
// rust: LIGHT_RECORD_STRIDE = 32 bytes
struct LightRecord {
    vec4 sphere;
    // packed normal, flux bits (luminance of L A, 0 = not a light), generation, lit_frame
    uvec4 info;
};
layout(set = SURFEL_SET, binding = 27, std430) buffer LightRecords { LightRecord light_rec[]; };

LightRecord empty_light_record() {
    LightRecord rec;
    rec.sphere = vec4(0.0);
    rec.info = uvec4(0u, 0u, 0u, INVALID_ID);
    return rec;
}

layout(set = SURFEL_SET, binding = 19, r32ui) uniform uimage2D surfel_overlay;

#define IMG_EIND 0
#define IMG_EDIR 1
#define IMG_EREF 2
#define IMG_EIND_RAW 3
#define IMG_EIND_HISTORY 4
#define IMG_COVERAGE 5
#define IMG_ALBEDO 6
#define GI_IMAGE_COUNT 7
layout(set = SURFEL_SET, binding = 20, rgba32f) uniform image2D gi_images[GI_IMAGE_COUNT];

// ---- pool --------------------------------------------------------------

uint alive_in_base() {
    return (gi_data.frame_index & 1u) * MAX_SURFELS;
}

uint alive_out_base() {
    return ((gi_data.frame_index + 1u) & 1u) * MAX_SURFELS;
}

// Pops a free slot. Only valid in passes that never push: once one pop fails
// the stack is empty for the rest of the pass, so successful pops see
// distinct counter values.
uint surfel_alloc() {
    int prev = atomicAdd(state[ST_FREE], -1);
    if (prev <= 0) {
        atomicAdd(state[ST_FREE], 1);
        atomicAdd(state[ST_SPAWN_FAILED], 1);
        return INVALID_ID;
    }
    return free_ids[prev - 1];
}

void surfel_push_alive(uint id) {
    uint index = uint(atomicAdd(state[ST_ALIVE_OUT], 1));
    alive_ids[alive_out_base() + index] = id;
}

mat4 instance_model(uint inst, uint bone) {
    if (inst >= NODE_CAPACITY) {
        return mat4(1.0);
    }
    if (bone == BONE_RIGID) {
        return node_world.world[inst];
    }
    uint joint = instance_bone_base.bone_base[inst] + bone;
    return joint < MAX_PALETTE_BONES ? bone_palette.bone[joint] : mat4(1.0);
}

// Creates a surfel at world position p with normal n. The slot must come from
// surfel_alloc(); the surfel joins this frame's alive list.
void surfel_init(uint id, uint inst, vec3 p, vec3 n, vec3 albedo, float radius, vec3 e_dir, float e_dir_n) {
    mat4 model = instance_model(inst, BONE_RIGID);
    mat4 inv = inverse(model);
    vec3 bind_p = (inv * vec4(p, 1.0)).xyz;
    vec3 bind_n = normalize(transpose(mat3(model)) * n);
    radius = clamp(radius, SURFEL_MIN_RADIUS, SURFEL_MAX_RADIUS);

    Surfel s;
    s.position = p;
    s.radius = radius;
    s.bind_position = bind_p;
    s.instance_id = inst;
    s.normal_bind = compress_unit_vec(bind_n);
    s.bone = BONE_RIGID;
    s.instance_generation = inst < NODE_CAPACITY ? instance_generation.generation[inst] : 0u;
    s.generation = surfels[id].generation + 1u;
    s.albedo = albedo;
    s.area = PI * radius * radius;
    s.e_dir = e_dir;
    s.e_dir_n = e_dir_n;
    s.e_ind = vec3(0.0);
    s.e_ind_n = 0.0;
    s.normal = n;
    s.flags = SURFEL_FLAG_ALIVE;
    s.last_seen = gi_data.frame_index;
    s.area_n = 0.0;
    s.spawn_frame = gi_data.frame_index;
    s.lit_frame = INVALID_ID;
    surfels[id] = s;

    geo[id] = vec4(p, radius);
    nrm[id] = compress_unit_vec(n);
    // Not a light until the publish pass computes its area-weighted radiance.
    surfel_light[id] = vec4(0.0);
    light_rec[id] = empty_light_record();
    surfel_push_alive(id);
}

// ---- hashed grid -------------------------------------------------------

float grid_cell_size(uint level) {
    return GRID_BASE_CELL * float(1u << level);
}

// Smallest level whose cell is at least the surfel diameter. rust: grid_level
uint grid_level(float radius) {
    uint level = 0u;
    while (level + 1u < GRID_LEVELS && grid_cell_size(level) < 2.0 * radius) {
        level++;
    }
    return level;
}

// rust: grid_bucket
uint grid_bucket(ivec3 cell, uint level) {
    uint h = hash_u32(level + 0x632be5abu);
    h = hash_u32(h ^ uint(cell.x));
    h = hash_u32(h ^ uint(cell.y));
    h = hash_u32(h ^ uint(cell.z));
    return h & (GRID_BUCKETS - 1u);
}

// Distinct buckets of the (at most 2x2x2) cells the surfel's bounding box
// overlaps on its level.
uint surfel_buckets(vec3 c, float r, out uint buckets[8]) {
    uint level = grid_level(r);
    float cs = grid_cell_size(level);
    ivec3 lo = ivec3(floor((c - vec3(r)) / cs));
    ivec3 hi = min(ivec3(floor((c + vec3(r)) / cs)), lo + ivec3(1));
    uint count = 0u;
    for (int z = lo.z; z <= hi.z; z++) {
        for (int y = lo.y; y <= hi.y; y++) {
            for (int x = lo.x; x <= hi.x; x++) {
                uint b = grid_bucket(ivec3(x, y, z), level);
                bool seen = false;
                for (uint k = 0u; k < count; k++) {
                    seen = seen || buckets[k] == b;
                }
                if (!seen) {
                    buckets[count] = b;
                    count++;
                }
            }
        }
    }
    return count;
}

// Distinct buckets a point query must read: one cell per occupied level.
// Every surfel whose sphere contains p appears exactly once in their union.
uint point_buckets(vec3 p, out uint buckets[GRID_LEVELS]) {
    uint mask = uint(state[ST_LEVEL_MASK]);
    uint count = 0u;
    for (uint level = 0u; level < GRID_LEVELS; level++) {
        if ((mask & (1u << level)) == 0u) {
            continue;
        }
        uint b = grid_bucket(ivec3(floor(p / grid_cell_size(level))), level);
        bool seen = false;
        for (uint k = 0u; k < count; k++) {
            seen = seen || buckets[k] == b;
        }
        if (!seen) {
            buckets[count] = b;
            count++;
        }
    }
    return count;
}

float surfel_kernel_id(uint id, vec3 p, vec3 np) {
    vec4 g = geo[id];
    return surfel_kernel(p, np, g.xyz, decompress_unit_vec(nrm[id]), g.w);
}

// ---- surfels as lights --------------------------------------------------

// Equal-area polygon standing in for a disk cut by a receiver's horizon.
#define DISK_POLY_N 16
// sqrt(2 pi / (N sin(2 pi / N))): circumradius of the N-gon with the disk's area
#define DISK_POLY_SCALE 1.0130073

// Geometry term G of a uniform Lambertian disk (center c, normal nc, radius
// r) seen from (x, nx): irradiance = L A G, the emitted power L A being spread
// over the disk. rust: disk_geometry
//  - disk entirely above the receiver's horizon: the disk's vector
//    irradiance, exact. With T = |x - c|^2 + r^2, S = sqrt(T^2 - 4 rho^2 r^2),
//    h the height over the disk plane and rho the in-plane offset:
//      G = -((nx.nc)(S + T - 2 rho^2) + 2 h (nx.rho)) / (S (S + T))
//  - disk cut by the horizon (concave corners: a lit ledge under a wall): the
//    equal-area 16-gon clipped to the horizon, integrated exactly with
//    Lambert's polygon formula.
// Whether part of the disk is in front of the disk and above the receiver's
// horizon: exactly where disk_geometry() is positive. Inside the support the
// geometry term is at least DISK_SUPPORT_FLOOR (the clipped polygon can miss a
// sliver of the disk), so ReSTIR's Z can use this test instead of the target.
#define DISK_SUPPORT_FLOOR 1e-30
bool disk_supports(vec3 c, vec3 nc, float r, vec3 x, vec3 nx) {
    const vec3 dv = x - c;
    if (dot(dv, nc) <= 0.0 || r <= 0.0) {
        return false;
    }
    const float extent = r * sqrt(max(1.0 - dot(nc, nx) * dot(nc, nx), 0.0));
    return -dot(dv, nx) > -extent;
}

float disk_geometry(vec3 c, vec3 nc, float r, vec3 x, vec3 nx) {
    if (!disk_supports(c, nc, r, x, nx)) {
        return 0.0;
    }
    const vec3 dv = x - c;
    const float h = dot(dv, nc);
    const float above = -dot(dv, nx);
    const float extent = r * sqrt(max(1.0 - dot(nc, nx) * dot(nc, nx), 0.0));
    if (above >= extent) {
        const vec3 rv = dv - h * nc;
        const float rho2 = dot(rv, rv);
        const float t = dot(dv, dv) + r * r;
        const float s = sqrt(max(t * t - 4.0 * rho2 * r * r, 0.0));
        const float denom = s * (s + t);
        if (!(denom > 0.0)) {
            return DISK_SUPPORT_FLOOR;
        }
        const float num = dot(nx, nc) * (s + t - 2.0 * rho2) + 2.0 * h * dot(nx, rv);
        return max(-num / denom, DISK_SUPPORT_FLOOR);
    }

    vec3 tb, bb;
    make_basis(nc, tb, bb);
    const float rp = r * DISK_POLY_SCALE;
    vec3 poly[DISK_POLY_N + 1];
    uint count = 0u;
    vec3 a = c + rp * tb;
    float da = dot(a - x, nx);
    for (uint k = 0u; k < DISK_POLY_N; k++) {
        const float phi = 2.0 * PI * float(k + 1u) / float(DISK_POLY_N);
        const vec3 b = c + rp * (cos(phi) * tb + sin(phi) * bb);
        const float db = dot(b - x, nx);
        if (da >= 0.0) {
            poly[count++] = a;
        }
        if ((da >= 0.0) != (db >= 0.0)) {
            poly[count++] = a + (b - a) * (da / (da - db));
        }
        a = b;
        da = db;
    }
    float sum = 0.0;
    for (uint k = 0u; k < count; k++) {
        const vec3 ui = normalize(poly[k] - x);
        const vec3 uj = normalize(poly[(k + 1u) % count] - x);
        const vec3 g = cross(ui, uj);
        const float gl = length(g);
        if (gl > 1e-12) {
            sum += acos(clamp(dot(ui, uj), -1.0, 1.0)) * dot(nx, g) / gl;
        }
    }
    return max(abs(sum) * 0.5 / (PI * r * r), DISK_SUPPORT_FLOOR);
}

// Unshadowed irradiance at (x, nx) from surfel j as a Lambertian disk light
// (the disk of its light record).
vec3 surfel_irradiance(uint id, vec3 x, vec3 nx) {
    const vec4 la = surfel_light[id];
    if (la.w <= 0.0) {
        return vec3(0.0);
    }
    const LightRecord rec = light_rec[id];
    return la.rgb * disk_geometry(rec.sphere.xyz, decompress_unit_vec(rec.info.x), rec.sphere.w, x, nx);
}

// Unshadowed irradiance luminance from a disk light with flux = luminance(L A).
// Equals luminance(surfel_irradiance()).
float disk_target(vec4 sphere, uint packed_n, float flux, vec3 x, vec3 nx) {
    if (!(flux > 0.0)) {
        return 0.0;
    }
    return flux * disk_geometry(sphere.xyz, decompress_unit_vec(packed_n), sphere.w, x, nx);
}

float record_target(LightRecord rec, vec3 x, vec3 nx) {
    return disk_target(rec.sphere, rec.info.x, uintBitsToFloat(rec.info.y), x, nx);
}

// Whether record_target() is positive; the Z of ReSTIR only needs this.
bool record_supports(LightRecord rec, vec3 x, vec3 nx) {
    return uintBitsToFloat(rec.info.y) > 0.0
        && disk_supports(rec.sphere.xyz, decompress_unit_vec(rec.info.x), rec.sphere.w, x, nx);
}

float surfel_target(uint id, vec3 x, vec3 nx) {
    return record_target(light_rec[id], x, nx);
}


// Outgoing radiance of the surfel cache at (p, n): kernel-weighted mean of
// the covering surfels' L over those already published as lights (w_lit is
// their weight sum). w_sum counts every covering surfel; `dominant` is the
// surfel with the largest weight.
vec3 cache_radiance(vec3 p, vec3 n, out float w_sum, out float w_lit, out uint dominant) {
    uint buckets[GRID_LEVELS];
    uint count = point_buckets(p, buckets);
    vec3 sum = vec3(0.0);
    w_sum = 0.0;
    w_lit = 0.0;
    dominant = INVALID_ID;
    float best = 0.0;
    for (uint k = 0u; k < count; k++) {
        uint b = buckets[k];
        uint end = grid_start[b + 1u];
        for (uint i = grid_start[b]; i < end; i++) {
            uint id = grid_items[i];
            float w = surfel_kernel_id(id, p, n);
            if (w <= 0.0) {
                continue;
            }
            vec4 la = surfel_light[id];
            if (la.w > 0.0) {
                sum += w * (la.rgb / la.w);
                w_lit += w;
            }
            w_sum += w;
            if (w > best) {
                best = w;
                dominant = id;
            }
        }
    }
    return w_lit > 0.0 ? sum / w_lit : vec3(0.0);
}

#endif // _GI_SURFELS_
