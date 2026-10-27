#ifndef _AO_
#define _AO_

// World-space radius of the neighborhood that can occlude a pixel.
// Both algorithms use this length, so a larger world does not change the
// physical size of the effect. 1 in the AO image is fully visible.
#define AO_RADIUS 64.0
// Occluders closer than this to the tangent plane are treated as the surface itself.
#define AO_BIAS 1.0
#define AO_SAMPLE_COUNT 16
// Occlusion below this is depth noise on open surfaces. What remains is scaled
// so a crease, not a flat wall, is what goes dark.
#define AO_SSAO_FLOOR 0.025
#define AO_SSAO_SCALE 3.5
#define RTAO_RAYS 16
#define AO_BLUR_RADIUS 4
#define AO_BLUR_SIGMA 2.0
// Bilateral weights. The plane test scales with distance from the eye, same
// threshold the GI reprojection uses, so a far surface does not reject
// neighbors that are a fraction of a pixel apart.
#define AO_BLUR_NORMAL_COS 0.8
#define AO_BLUR_PLANE 0.02

#endif // _AO_
