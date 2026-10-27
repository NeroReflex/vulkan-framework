#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT float ao_hit_t;

void main() {
    ao_hit_t = gl_HitTEXT;
}
