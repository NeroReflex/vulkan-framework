// ============================================ FRAGMENT OUTPUT ==================================================
layout (location = 0) out vec4 out_vPosition;           // Search for GBUFFER_FB0
layout (location = 1) out vec4 out_vNormal;             // Search for GBUFFER_FB1
layout (location = 2) out vec4 out_vDiffuse;            // Search for GBUFFER_FB2
layout (location = 3) out vec4 out_vSpecular;           // Search for GBUFFER_FB3
layout (location = 4) out vec4 out_vInstanceId;         // Search for GBUFFER_FB4
// ===============================================================================================================

layout (location = 0) in vec4 in_vPosition_worldspace;
layout (location = 1) in vec4 in_vNormal_worldspace;
layout (location = 2) in vec2 in_vTextureUV;
layout (location = 3) in vec4 in_vPosition_worldspace_minus_eye_position;
layout (location = 4) in flat vec4 in_eyePosition_worldspace;

layout(push_constant) uniform MeshData {
    mat3x4 load_matrix;
    uint mesh_id;
} mesh_data;

// MUST match with MAX_TEXTURES on rust side
layout(set = 0, binding = 0) uniform sampler2D textures[256];

struct material_t {
    uint diffuse_texture_index;
    uint normal_texture_index;
    uint reflection_texture_index;
    uint displacement_texture_index;
};

struct mesh_to_material_t {
    uint material_index;
};

layout(std430, set = 1, binding = 0) readonly buffer material
{
    material_t info[];
};

layout(std430, set = 1, binding = 1) readonly buffer meshes
{
    mesh_to_material_t material_for_mesh[];
};

void main() {
    // Calculate position of the current fragment
    //vec4 vPosition_worldspace = vec4((in_vPosition_worldspace_minus_eye_position + in_eyePosition_worldspace).xyz, 1.0);

    vec3 dFdxPos = dFdx( in_vPosition_worldspace_minus_eye_position.xyz );
	vec3 dFdyPos = dFdy( in_vPosition_worldspace_minus_eye_position.xyz );
	const vec3 facenormal = cross(dFdxPos, dFdyPos);

    // The normal can either be calculated or provided from the mesh. Just pick the provided one if it is valid.
    const vec3 bestNormal = normalize(length(in_vNormal_worldspace.xyz) < 0.000001f ? facenormal : in_vNormal_worldspace.xyz);

    const uint material_id = material_for_mesh[mesh_data.mesh_id].material_index;
    const uint diffuse_texture_index = info[material_id].diffuse_texture_index;

    // in OpenGL depth is in range [-1;+1], while in vulkan it is [0.0;1.0]
    // see https://docs.vulkan.org/guide/latest/depth.html "Porting from OpenGL"
    //vPosition_worldspace.z = (vPosition_worldspace.z + vPosition_worldspace.w) * 0.5;

    out_vPosition = in_vPosition_worldspace;
    out_vNormal = vec4(bestNormal.xyz, 0.0);
    out_vDiffuse = texture(textures[diffuse_texture_index], in_vTextureUV);
    out_vSpecular = vec4(0.0, 0.0, 0.0, 0.0);
    out_vInstanceId = vec4(float(mesh_data.mesh_id), 0.0, 0.0, 0.0);
}
