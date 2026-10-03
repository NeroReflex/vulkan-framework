#ifndef _CONFIG_
#define _CONFIG_

// keep in sync with sources
#define MAX_DIRECTIONAL_LIGHTS 8

#define ENABLE_SURFELS 1

// this is for difficult scenes where once a path of light
// is found, it is very unlikely that new paths will be found
// or the same path is (quickly) found again.
//
// Enabling this vastly increases the memory pressure on the same
// memory area, accessed with atomic updates: a very expensive operation.
#define FORCE_ALLOCATION 0

// Keep in sync wit rust side
#define MAX_SURFELS_PER_FRAME 256

// If this is enabled surfel that haven't contributed to the final image
// in N frames are deleted
#define DELETE_NOT_CONTRIBUTING_SURFELS 120

#define MIN_SURFEL_RADIUS 10.0
#define MAX_SURFEL_RADIUS 120.0
#define DELETE_ON_RADIUS_DIFFERENCE 15.0

#define POPULATE_VISIBLE_SURFELS_LIST 0

#define SHOW_SURFELS 1

#define SURFEL_IMPORTANCE_SAMPLES 4

#define CLOSEST_INTERSECTION_DISTANCE 0.4

// This MUST be kept in sync with rust side
#define MAX_USABLE_SURFELS 8182

#define USED_SURFEL_MISSING 0xFFFFFFFFu

#define VIRTUAL_POINT_LIGHTS_PER_PIXEL 4u

#define MAX_BVH_STACK_DEPTH 96

// Discovery must cover every pixel (surfel_rt reads overlapping per texel).
#define SURFEL_DISCOVERY_QUERY_STRIDE 1u
// VPL may use a coarser stride (2 = quarter dispatch count).
#define SURFEL_VPL_QUERY_STRIDE 2u

// Set to 1 only to time the surfel index without software ray tracing.
// Keep 0 unless you are actively debugging a shader: enabling the
// GL_EXT_debug_printf extension makes GPU-AV instrument the pass, which
// currently TDR's this pipeline (millions of invocations per frame).
#ifndef ENABLE_DEBUG_PRINTF
#define ENABLE_DEBUG_PRINTF 0
#endif

#endif // _CONFIG_
