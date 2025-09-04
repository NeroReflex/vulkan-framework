#ifndef _TEXTURES_
#define _TEXTURES_

#ifndef TEXTURES_DESCRIPTOR_SET
#define TEXTURES_DESCRIPTOR_SET 3
#endif

// MUST match with MAX_TEXTURES on rust side
layout(set = TEXTURES_DESCRIPTOR_SET, binding = 0) uniform sampler2D textures[256];

#endif
