
#ifndef _PAYLOAD_
#define _PAYLOAD_

struct hit_payload_t {
    bool hit;
    uint instance_id;
    vec3 position;
    vec3 triangle_normal;
    vec3 diffuse;
};

#endif // _PAYLOAD_
