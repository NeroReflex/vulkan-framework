#include "../config.glsl"

layout (location = 0) out vec4 outColor;

layout (location = 0) in vec2 in_vTextureUV;

#define GBUFFER_DESCRIPTOR_SET 0
#include "../gbuffer.glsl"

layout(set = 1, binding = 0) uniform sampler2D gibuffer[2];

#define STATUS_DESCRIPTOR_SET 2
#include "../status.glsl"

void main() {
    const vec3 in_vPosition_worldspace = texture(gbuffer[0], in_vTextureUV).xyz;
    const vec3 in_vNormal_worldspace = texture(gbuffer[1], in_vTextureUV).xyz;
    const vec4 in_vDiffuseAlbedo = texture(gbuffer[2], in_vTextureUV);

    const vec3 global_illumination_received = texture(gibuffer[0], in_vTextureUV).xyz;
    const vec3 directional_light_received = texture(gibuffer[1], in_vTextureUV).xyz;

    const vec3 mc_gi_contribution = global_illumination_received / float(gi_data.gi_reuse_frames + 1u);

    vec3 out_vDiffuseAlbedo = directional_light_received + mc_gi_contribution;
    #if SHOW_SURFELS
        out_vDiffuseAlbedo = directional_light_received * 0.1 + mc_gi_contribution;
    #endif

    outColor = vec4(out_vDiffuseAlbedo.xyz, 1.0);
}
