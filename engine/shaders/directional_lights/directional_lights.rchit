#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require

// just a stub
layout(std430, set = 0, binding = 0) readonly buffer tlas_instances
{
    uint data[];
};

layout(location = 0) rayPayloadInEXT bool hitValue;

//hitAttributeEXT vec2 attribs;

void main() {
    hitValue = true;
    //const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    //hitValue = barycentricCoords;
}
