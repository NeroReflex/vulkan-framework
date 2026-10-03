#ifndef _SURFEL_BUILD_LAYOUT_
#define _SURFEL_BUILD_LAYOUT_

// Must match the BUILD_* constants in global_illumination.rs
#define BUILD_HALF 32768u
#define BUILD_BLOCKS (BUILD_HALF / 256u)
#define RADIX_BINS 256u
#define KEY_STRIDE 2u

#define HDR_HOLE_COUNT 1u
#define HDR_ROOT 2u
#define HDR_KEY_COUNT 3u
#define HDR_HEAP_P 4u
#define HDR_HEAP_LEAF0 5u
#define HDR_BUILD_ACTIVE 7u

#define OFF_BLOCK_SUMS 8u
#define OFF_BLOCK_EXCL (OFF_BLOCK_SUMS + BUILD_BLOCKS)
#define OFF_HOLES (OFF_BLOCK_EXCL + BUILD_BLOCKS)
#define OFF_IDS_A (OFF_HOLES + BUILD_HALF)

#define OFF_KEYS_A (OFF_IDS_A + BUILD_HALF)
#define OFF_KEYS_B (OFF_KEYS_A + BUILD_HALF * KEY_STRIDE)
#define OFF_LEAF_PARENT (OFF_KEYS_B + BUILD_HALF * KEY_STRIDE)
#define OFF_RADIX_HIST (OFF_LEAF_PARENT + BUILD_HALF)
#define OFF_RADIX_EXCL (OFF_RADIX_HIST + BUILD_BLOCKS * RADIX_BINS)
#define OFF_DIGIT_BASE (OFF_RADIX_EXCL + BUILD_BLOCKS * RADIX_BINS)
// Per-block then global AABB of live surfel centers, as ordered floats.
// Morton codes are quantized in this box. The camera far plane is ~1e4,
// and a 10-bit code over that cube is a ~20-unit cell: every surfel in
// the atrium shares a handful of keys and the tree cannot cull.
#define OFF_BLOCK_AABB (OFF_DIGIT_BASE + RADIX_BINS)
#define OFF_AABB (OFF_BLOCK_AABB + BUILD_BLOCKS * 6u)
#define BUILD_WORDS (OFF_AABB + 6u)

// Spatial hash over live-list slots. Heads live in the unused key region.
// One node per live surfel (OFF_HASH_NEXT / OFF_HASH_IDS indexed by live slot).
#define HASH_LEVELS 3u
#define HASH_BUCKETS 8192u
#define HASH_EMPTY 0xFFFFFFFFu
#define OFF_HASH_HEAD OFF_KEYS_A
#define OFF_HASH_NEXT (OFF_HASH_HEAD + HASH_LEVELS * HASH_BUCKETS)
#define OFF_HASH_IDS (OFF_HASH_NEXT + BUILD_HALF)

#endif
