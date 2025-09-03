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

#define SHOW_SURFELS 1

#define SURFEL_IMPORTANCE_SAMPLES 4

#define CLOSEST_INTERSECTION_DISTANCE 0.4

// This MUST be kept in sync with rust side
#define MAX_USABLE_SURFELS 8182

#define USED_SURFEL_MISSING 0xFFFFFFFFu

#define VIRTUAL_POINT_LIGHTS_PER_PIXEL 4u

#endif // _CONFIG_
