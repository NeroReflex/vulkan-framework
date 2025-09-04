#ifndef _MATERIALS_
#define _MATERIALS_

#ifndef MATERIALS_DESCRIPTOR_SET
#define MATERIALS_DESCRIPTOR_SET 1
#endif

struct material_t {
    uint diffuse_texture_index;
    uint normal_texture_index;
    uint reflection_texture_index;
    uint displacement_texture_index;
};

struct mesh_to_material_t {
    uint material_index;
};

layout(std430, set = MATERIALS_DESCRIPTOR_SET, binding = 0) readonly buffer material
{
    material_t info[];
};

layout(std430, set = MATERIALS_DESCRIPTOR_SET, binding = 1) readonly buffer meshes
{
    mesh_to_material_t material_for_mesh[];
};

#endif // _MATERIALS_
