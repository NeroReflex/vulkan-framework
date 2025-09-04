// ============================================ FRAGMENT OUTPUT ==================================================
layout (location = 0) out vec4 out_vPosition;           // Search for GBUFFER_FB0
layout (location = 1) out vec4 out_vNormal;             // Search for GBUFFER_FB1
layout (location = 2) out vec4 out_vDiffuse;            // Search for GBUFFER_FB2
layout (location = 3) out vec4 out_vSpecular;           // Search for GBUFFER_FB3
layout (location = 4) out uvec4 out_vInstanceId;         // Search for GBUFFER_FB4
// ===============================================================================================================

layout (location = 0) in vec4 in_vPosition_worldspace;
layout (location = 1) in vec4 in_vNormal_worldspace;
layout (location = 2) in vec2 in_vTextureUV;
layout (location = 3) in vec4 in_vPosition_worldspace_minus_eye_position;
layout (location = 4) in flat vec4 in_eyePosition_worldspace;
layout (location = 5) in flat uint offset_instance_id;

layout(push_constant) uniform MeshData {
    mat3x4 load_matrix;
    uint mesh_id;
    uint base_instance;
} mesh_data;

#define TEXTURES_DESCRIPTOR_SET 0
#include "../textures.glsl"

#define MATERIALS_DESCRIPTOR_SET 1
#include "../materials.glsl"

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
    out_vInstanceId = uvec4(mesh_data.mesh_id, mesh_data.base_instance + offset_instance_id, 0, 0);
}
