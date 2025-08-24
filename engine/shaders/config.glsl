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

#define MAX_SURFELS_PER_FRAME 64

#define MAX_SURFEL_RADIUS 20.0

#define SHOW_SURFELS 0

#endif // _CONFIG_
