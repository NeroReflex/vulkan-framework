//#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
//#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
//#extension GL_EXT_shader_explicit_arithmetic_types_int64 : required

#extension GL_GOOGLE_include_directive : enable


#include "payload.glsl"

#define Buffer(Alignment) \
  layout(buffer_reference, std430, buffer_reference_align = Alignment) buffer

struct vertex_buffer_element_t {
    float position_x;
    float position_y;
    float position_z;
    float normal_x;
    float normal_y;
    float normal_z;
    float texture_u;
    float texture_v;
};

Buffer(32) VertexBuffer {
  vertex_buffer_element_t vertex_data[ /* address by IndexBuffer */ ];
};

Buffer(4) IndexBuffer {
  uint vertex_index[ /* address by gl_PrimitiveID */ ];
};

Buffer(32) TransformBuffer {
  mat3x4 transform;
};

struct blas_data_t {
    mat3x4 model_matrix;
    uint instance_shader_binding_table_record_offset_and_flags;
    uint instance_custom_index_and_mask;
    uint acceleration_structure_reference_1;
    uint acceleration_structure_reference_2;
};

Buffer(64) InstanceBuffer {
    blas_data_t blas_ref[];
};

struct tlas_instance_data_t {
    IndexBuffer ib;
    VertexBuffer vb;
    TransformBuffer tb;
    InstanceBuffer instance;
};

layout(std430, set = 0, binding = 0) readonly buffer tlas_instances
{
    tlas_instance_data_t data[ /* address by gl_InstanceID */ ];
};

// MUST match with MAX_TEXTURES on rust side
layout(set = 3, binding = 0) uniform sampler2D textures[256];

struct material_t {
    uint diffuse_texture_index;
    uint normal_texture_index;
    uint reflection_texture_index;
    uint displacement_texture_index;
};

layout(std430, set = 4, binding = 0) readonly buffer material
{
    material_t info[];
};

struct mesh_to_material_t {
    uint material_index;
};

layout(std430, set = 4, binding = 1) readonly buffer meshes
{
    mesh_to_material_t material_for_mesh[];
};

layout(location = 0) rayPayloadInEXT hit_payload_t payload;

hitAttributeEXT vec2 attribs;

/*
 * gl_InstanceCustomIndexEXT => ci ho messo il numero della mesh per ricondurmi al materiale giusto
 * gl_InstanceID => il numero della istanza: ogni istanza ha un numero progressivo che parte da 0
 * gl_PrimitiveID => l'indice del triangolo colpito
 */

vec3 calculateNormal(vec3 A, vec3 B, vec3 C) {
    vec3 edge1 = B - A;
    vec3 edge2 = C - A;
    vec3 normal = cross(edge1, edge2);
    return normalize(normal);
}

void main() {
    payload.hit = true;

    const uint first_vertex_id = gl_PrimitiveID * 3;

    const uvec3 vertex_index = uvec3(
        data[gl_InstanceID].ib.vertex_index[first_vertex_id + 0],
        data[gl_InstanceID].ib.vertex_index[first_vertex_id + 1],
        data[gl_InstanceID].ib.vertex_index[first_vertex_id + 2]
    );
    const vec4 v1_position = vec4(
        data[gl_InstanceID].vb.vertex_data[vertex_index.x].position_x,
        data[gl_InstanceID].vb.vertex_data[vertex_index.x].position_y,
        data[gl_InstanceID].vb.vertex_data[vertex_index.x].position_z,
        1.0
    );
    const vec4 v2_position = vec4(
        data[gl_InstanceID].vb.vertex_data[vertex_index.y].position_x,
        data[gl_InstanceID].vb.vertex_data[vertex_index.y].position_y,
        data[gl_InstanceID].vb.vertex_data[vertex_index.y].position_z,
        1.0
    );
    const vec4 v3_position = vec4(
        data[gl_InstanceID].vb.vertex_data[vertex_index.z].position_x,
        data[gl_InstanceID].vb.vertex_data[vertex_index.z].position_y,
        data[gl_InstanceID].vb.vertex_data[vertex_index.z].position_z,
        1.0
    );

    const vec4 v1_normal = vec4(
        data[gl_InstanceID].vb.vertex_data[vertex_index.x].normal_x,
        data[gl_InstanceID].vb.vertex_data[vertex_index.x].normal_y,
        data[gl_InstanceID].vb.vertex_data[vertex_index.x].normal_z,
        0.0
    );
    const vec4 v2_normal = vec4(
        data[gl_InstanceID].vb.vertex_data[vertex_index.y].normal_x,
        data[gl_InstanceID].vb.vertex_data[vertex_index.y].normal_y,
        data[gl_InstanceID].vb.vertex_data[vertex_index.y].normal_z,
        0.0
    );
    const vec4 v3_normal = vec4(
        data[gl_InstanceID].vb.vertex_data[vertex_index.z].normal_x,
        data[gl_InstanceID].vb.vertex_data[vertex_index.z].normal_y,
        data[gl_InstanceID].vb.vertex_data[vertex_index.z].normal_z,
        0.0
    );

    const vec2 v1_uv = vec2(
        data[gl_InstanceID].vb.vertex_data[vertex_index.x].texture_u,
        data[gl_InstanceID].vb.vertex_data[vertex_index.x].texture_v
    );
    const vec2 v2_uv = vec2(
        data[gl_InstanceID].vb.vertex_data[vertex_index.y].texture_u,
        data[gl_InstanceID].vb.vertex_data[vertex_index.y].texture_v
    );
    const vec2 v3_uv = vec2(
        data[gl_InstanceID].vb.vertex_data[vertex_index.z].texture_u,
        data[gl_InstanceID].vb.vertex_data[vertex_index.z].texture_v
    );

    const mat3x4 load_matrix = data[gl_InstanceID].tb.transform;
    const mat3x4 transform_matrix = data[gl_InstanceID].instance.blas_ref[gl_InstanceID].model_matrix;

    const mat4 model_matrix =
        mat4(transform_matrix[0], transform_matrix[1], transform_matrix[2], vec4(0.0, 0.0, 0.0, 1.0)) *
        mat4(load_matrix[0], load_matrix[1], load_matrix[2], vec4(0.0, 0.0, 0.0, 1.0));

    const uint mesh_id = gl_InstanceCustomIndexEXT;

    //const mat4 model_matrix = mat4(gl_ObjectToWorldEXT[0], gl_ObjectToWorldEXT[1], gl_ObjectToWorldEXT[2], vec4(0.0, 0.0, 0.0, 1.0));

    vec4 v1_world_position = model_matrix * v1_position;
    v1_world_position /= v1_world_position.w;
    vec4 v2_world_position = model_matrix * v2_position;
    v2_world_position /= v2_world_position.w;
    vec4 v3_world_position = model_matrix * v3_position;
    v3_world_position /= v3_world_position.w;

    const vec4 v1_world_normal = model_matrix * v1_normal;
    const vec4 v2_world_normal = model_matrix * v2_normal;
    const vec4 v3_world_normal = model_matrix * v3_normal;

    const uint material_index = material_for_mesh[nonuniformEXT(mesh_id)].material_index;
    const uint diffuse_texture_id = info[nonuniformEXT(material_index)].diffuse_texture_index;

    const vec3 barycentrics = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);

    vec2 texture_uv = v1_uv * barycentrics.x + v2_uv * barycentrics.y + v3_uv * barycentrics.z;

    const vec4 diffuse_surface_color = texture(textures[nonuniformEXT(diffuse_texture_id)], texture_uv);

    payload.position = v1_world_position.xyz * barycentrics.x + v2_world_position.xyz * barycentrics.y + v3_world_position.xyz * barycentrics.z;
    payload.triangle_normal = calculateNormal(v1_world_normal.xyz, v2_world_normal.xyz, v3_world_normal.xyz);
    payload.diffuse = diffuse_surface_color.xyz;
    payload.instance_id = gl_InstanceID;
}
