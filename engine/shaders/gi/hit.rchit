#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable

#include "payload.glsl"
#include "../row3x4.glsl"

#define RT_DESCRIPTOR_SET 0
#include "../rt_descriptor_set.glsl"

#define TEXTURES_DESCRIPTOR_SET 4
#include "../textures.glsl"

#define MATERIALS_DESCRIPTOR_SET 5
#include "../materials.glsl"

layout(location = 0) rayPayloadInEXT hit_payload_t payload;

hitAttributeEXT vec2 attribs;

void main() {
    const uint inst = gl_InstanceID;
    const uint first = gl_PrimitiveID * 3u;
    const uvec3 idx = uvec3(
        data[inst].ib.vertex_index[first + 0u],
        data[inst].ib.vertex_index[first + 1u],
        data[inst].ib.vertex_index[first + 2u]
    );
    const vertex_buffer_element_t v0 = data[inst].vb.vertex_data[idx.x];
    const vertex_buffer_element_t v1 = data[inst].vb.vertex_data[idx.y];
    const vertex_buffer_element_t v2 = data[inst].vb.vertex_data[idx.z];
    const vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    const mat3x4 load_rows = data[inst].tb.transform;
    const mat3x4 instance_rows = data[inst].instance.blas_ref[inst].model_matrix;
    const mat4 model =
        row_major_3x4(instance_rows[0], instance_rows[1], instance_rows[2]) *
        row_major_3x4(load_rows[0], load_rows[1], load_rows[2]);
    const mat3 normal_matrix = transpose(inverse(mat3(model)));

    const vec3 p0 = (model * vec4(v0.position_x, v0.position_y, v0.position_z, 1.0)).xyz;
    const vec3 p1 = (model * vec4(v1.position_x, v1.position_y, v1.position_z, 1.0)).xyz;
    const vec3 p2 = (model * vec4(v2.position_x, v2.position_y, v2.position_z, 1.0)).xyz;
    vec3 ng = cross(p1 - p0, p2 - p0);
    const float ng_len = length(ng);
    ng = ng_len > 0.0 ? ng / ng_len : -gl_WorldRayDirectionEXT;
    if (dot(ng, gl_WorldRayDirectionEXT) > 0.0) {
        ng = -ng;
    }

    vec3 ns = normal_matrix * (
        vec3(v0.normal_x, v0.normal_y, v0.normal_z) * bary.x +
        vec3(v1.normal_x, v1.normal_y, v1.normal_z) * bary.y +
        vec3(v2.normal_x, v2.normal_y, v2.normal_z) * bary.z
    );
    const float ns_len = length(ns);
    ns = ns_len > 0.0 ? ns / ns_len : ng;
    if (dot(ns, ng) < 0.0) {
        ns = -ns;
    }

    // The raster path flips v; sample the same texel here.
    vec2 uv = vec2(v0.texture_u, v0.texture_v) * bary.x +
              vec2(v1.texture_u, v1.texture_v) * bary.y +
              vec2(v2.texture_u, v2.texture_v) * bary.z;
    uv.y = 1.0 - uv.y;

    const uint mesh_id = gl_InstanceCustomIndexEXT;
    const uint material_index = material_for_mesh[nonuniformEXT(mesh_id)].material_index;
    const uint diffuse_texture_id = info[nonuniformEXT(material_index)].diffuse_texture_index;
    const vec3 albedo = textureLod(textures[nonuniformEXT(diffuse_texture_id)], uv, 0.0).rgb;

    payload.position = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    payload.hit = 1u;
    payload.normal = ns;
    payload.instance_id = inst;
    payload.albedo = albedo;
    payload.t = gl_HitTEXT;
}
