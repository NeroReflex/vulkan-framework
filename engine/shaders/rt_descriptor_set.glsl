#ifndef _RT_DESCRIPTOR_SET_
#define _RT_DESCRIPTOR_SET_ 1

#extension GL_EXT_scalar_block_layout : enable
//#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
//#extension GL_EXT_shader_explicit_arithmetic_types_int64 : required

#ifndef RT_DESCRIPTOR_SET
#define RT_DESCRIPTOR_SET 0
#endif

#define Buffer(Alignment) \
  layout(buffer_reference, std430, buffer_reference_align = Alignment) readonly buffer

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

layout(std430, set = RT_DESCRIPTOR_SET, binding = 0) readonly buffer tlas_instances
{
    tlas_instance_data_t data[ /* address by gl_InstanceID */ ];
};

layout (set = RT_DESCRIPTOR_SET, binding = 1) uniform accelerationStructureEXT topLevelAS;

#endif
