#ifndef _GBUFFER_
#define _GBUFFER_

#ifndef GBUFFER_DESCRIPTOR_SET
    #define GBUFFER_DESCRIPTOR_SET 1
#endif

layout (set = GBUFFER_DESCRIPTOR_SET, binding = 0) uniform sampler2DShadow gbuffer_depth;

layout (set = GBUFFER_DESCRIPTOR_SET, binding = 1) uniform sampler2D gbuffer_instance_id;

// gbuffer: 0 for position, 1 for normal, 2 for diffuse texture, 3 for specular texture
layout (set = GBUFFER_DESCRIPTOR_SET, binding = 2) uniform sampler2D gbuffer[4];

#endif // _GBUFFER_
