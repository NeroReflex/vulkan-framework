#ifndef _GI_PAYLOAD_
#define _GI_PAYLOAD_

struct hit_payload_t {
    vec3 position;
    uint hit;
    // shading normal, flipped to face the ray origin
    vec3 normal;
    uint instance_id;
    vec3 albedo;
    float t;
};

#endif // _GI_PAYLOAD_
