#ifndef _CONFIG_
#define _CONFIG_

// Every constant marked "rust" is mirrored in
// engine/src/rendering/pipeline/gi_math.rs and must stay identical.

// rust: MAX_DIRECTIONAL_LIGHTS
#define MAX_DIRECTIONAL_LIGHTS 8

// ---- surfel pool -------------------------------------------------------
// rust: MAX_SURFELS (power of two)
#define MAX_SURFELS 65536u
// Target surfel radius in pixels at spawn time.
#define SURFEL_SCREEN_RADIUS_PX 24.0
#define SURFEL_MIN_RADIUS 2.0
// Must stay <= GRID_BASE_CELL * 2^(GRID_LEVELS-1) / 2 so a surfel's AABB
// spans at most 2 cells per axis on its level.
#define SURFEL_MAX_RADIUS 256.0
// Surfels farther than this from the eye are evicted and never spawned.
#define SURFEL_MAX_DISTANCE 6000.0
// Frames without being the dominant surfel of any sampled pixel or ray hit.
#define SURFEL_UNSEEN_FRAMES 240u
#define SURFEL_PRESSURE_UNSEEN_FRAMES 30u
// Pool pressure: free slots below MAX_SURFELS / SURFEL_PRESSURE_DIVISOR.
#define SURFEL_PRESSURE_DIVISOR 8u
#define SURFEL_SCREEN_SPAWN_BUDGET 2048
#define SURFEL_RAY_SPAWN_BUDGET 256
// Extra path bounces from an uncovered indirect hit before truncating.
#define SURFEL_PATH_CONTINUATION 3u
// a ray-spawned surfel's radius is at most this fraction of the ray length
#define SURFEL_RAY_RADIUS_FRACTION 0.5
// a surfel touched by the coverage pass within this many frames counts as on screen
#define SURFEL_RECENT_FRAMES 2u
// A pixel spawns a surfel when its partition-of-unity weight sum is below this.
#define SURFEL_SPAWN_COVERAGE 0.5
// A pixel dooms its smallest surfel when the weight sum exceeds this.
#define SURFEL_OVER_COVERAGE 4.0
// A surfel this many times larger than the local target radius is doomed
// once the properly sized surfels around it cover the pixel.
#define SURFEL_OVERSIZE_FACTOR 2.5

// Kernel support: normals must agree and the point must be near the disk plane.
#define SURFEL_KERNEL_MIN_COS 0.5
#define SURFEL_KERNEL_PLANE_FRACTION 0.25

// Still frames after which a GI_FLAG_CONVERGE cache settles (cache_settled).
#define SURFEL_SETTLE_FRAMES 128u
// A settled cache stops spawning offscreen surfels after this many more frames.
#define SURFEL_SETTLE_SPAWN_FRAMES 256u

// EMA floors: the running mean is exact until 1/alpha samples, then forgets.
// A settled cache drops them.
#define EMA_MIN_ALPHA_DIRECT 0.05
#define EMA_MIN_ALPHA_INDIRECT 0.03
#define EMA_MIN_ALPHA_AREA 0.02

// ---- hashed grid -------------------------------------------------------
// rust: GRID_LEVELS, GRID_BASE_CELL, GRID_BUCKETS
#define GRID_LEVELS 8u
#define GRID_BASE_CELL 4.0
#define GRID_BUCKETS (1u << 18u)
#define GRID_MAX_ITEMS (MAX_SURFELS * 8u)
// Each grid scan workgroup covers this many buckets.
#define GRID_SCAN_BLOCK 1024u

// ---- light sampling ----------------------------------------------------
// rust: LIGHT_TILES, LIGHT_TILE_SIZE
#define LIGHT_TILES 128u
#define LIGHT_TILE_SIZE 256u
#define FLUX_SCAN_BLOCK 1024u
// MAX_SURFELS * FLUX_QUANT_MAX < 2^32 keeps the integer flux CDF exact.
#define FLUX_QUANT_MAX 65535u

// ---- ReSTIR ------------------------------------------------------------
#define RESTIR_INITIAL_CANDIDATES 8u
#define RESTIR_TEMPORAL_M_CAP 20.0
#define RESTIR_SPATIAL_NEIGHBORS 4u
#define RESTIR_SPATIAL_RADIUS 30.0
#define RESTIR_NORMAL_THRESHOLD 0.9
// Plane distance allowed for reuse, relative to the distance from the eye.
#define RESTIR_PLANE_THRESHOLD 0.02

// ---- accumulation ------------------------------------------------------
#define ACCUM_MAX_HISTORY 32.0
// Reference path tracer samples per pixel per frame.
#define REFERENCE_SPP 2u

// ---- rays --------------------------------------------------------------
// Offset along the normal for secondary ray origins, in world units.
#define RAY_NORMAL_OFFSET 0.25
#define RAY_T_MIN 0.01
#define RAY_T_MAX 100000.0

// ---- status flags (rust: GI_FLAG_*) ------------------------------------
#define GI_FLAG_MULTIBOUNCE (1u << 0u)
#define GI_FLAG_REFERENCE (1u << 1u)
#define GI_FLAG_TEMPORAL (1u << 2u)
#define GI_FLAG_SPATIAL (1u << 3u)
#define GI_FLAG_CAPTURE (1u << 4u)
#define GI_FLAG_RESET (1u << 5u)
#define GI_FLAG_SPAWN (1u << 6u)
#define GI_FLAG_ACCUMULATE (1u << 7u)
#define GI_FLAG_CONVERGE (1u << 8u)

#endif // _CONFIG_
