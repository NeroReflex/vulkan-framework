#define MAX_DIRECTIONAL_LIGHTS 8

layout (location = 0) out vec4 outColor;

layout (location = 0) in vec2 in_vTextureUV;

struct light_t {
    float direction_x;
    float direction_y;
    float direction_z;

    float intensity_x;
    float intensity_y;
    float intensity_z;
};

// gbuffer: 0 for position, 1 for normal, 2 for diffuse texture
layout(set = 0, binding = 0) uniform sampler2D gbuffer[3];
layout(set = 1, binding = 0) uniform sampler2D gibuffer[2];

void main() {
    const vec3 in_vPosition_worldspace = texture(gbuffer[0], in_vTextureUV).xyz;
    const vec3 in_vNormal_worldspace = texture(gbuffer[1], in_vTextureUV).xyz;
    const vec4 in_vDiffuseAlbedo = texture(gbuffer[2], in_vTextureUV);

    const vec3 global_illumination_received = texture(gibuffer[0], in_vTextureUV).xyz;
    const vec3 directional_light_received = texture(gibuffer[1], in_vTextureUV).xyz;

    vec3 out_vDiffuseAlbedo = directional_light_received;

    const vec3 global_illumination_contribution = in_vDiffuseAlbedo.xyz * global_illumination_received;

    outColor = vec4(out_vDiffuseAlbedo.xyz + global_illumination_contribution, 1.0);
}
