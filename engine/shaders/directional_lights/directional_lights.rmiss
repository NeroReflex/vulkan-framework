#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT bool hitValue;

void main() {
    hitValue = false;
}
