layout (location = 0) in vec3 vertex_position_modelspace;
layout (location = 1) in vec3 vertex_normal_modelspace;
layout (location = 2) in vec2 vertex_texture;

layout (location = 3) in vec4 ModelMatrix_first_row;
layout (location = 4) in vec4 ModelMatrix_second_row;
layout (location = 5) in vec4 ModelMatrix_third_row;

//layout (location = 3) in uint vMaterialIndex;

layout (location = 0) out vec4 out_vPosition_worldspace;
layout (location = 1) out vec4 out_vNormal_worldspace;
layout (location = 2) out vec2 out_vTextureUV;
layout (location = 3) out vec4 out_vPosition_worldspace_minus_eye_position;
layout (location = 4) out flat vec4 eyePosition_worldspace;

layout(std140, set = 2, binding = 0) uniform camera_uniform {
	mat4 viewMatrix;
	mat4 projectionMatrix;
} camera;

layout(push_constant) uniform MeshData {
    mat3x4 load_matrix;
    uint mesh_id;
} mesh_data;

void main() {
    const mat4 LoadMatrix = mat4(mesh_data.load_matrix[0], mesh_data.load_matrix[1], mesh_data.load_matrix[2], vec4(0.0, 0.0, 0.0, 1.0));
    const mat4 InstanceMatrix = mat4(ModelMatrix_first_row, ModelMatrix_second_row, ModelMatrix_third_row, vec4(0.0, 0.0, 0.0, 1.0));

    const mat4 ModelMatrix = InstanceMatrix * LoadMatrix;

    const mat4 MVP = camera.projectionMatrix * camera.viewMatrix * ModelMatrix; 

    // Get the eye position
	const vec4 eye_position = vec4(camera.viewMatrix[3][0], camera.viewMatrix[3][1], camera.viewMatrix[3][2], 1.0);
	eyePosition_worldspace = eye_position;

	vec4 vPosition_worldspace = ModelMatrix * vec4(vertex_position_modelspace, 1.0);
	vPosition_worldspace /= vPosition_worldspace.w;

    out_vPosition_worldspace = vPosition_worldspace;
	out_vTextureUV = vec2(vertex_texture.x, 1-vertex_texture.y);
	out_vPosition_worldspace_minus_eye_position = vec4((vPosition_worldspace - eyePosition_worldspace).xyz, 1.0);
	out_vNormal_worldspace = vec4((ModelMatrix * vec4(vertex_normal_modelspace, 0.0)).xyz, 0.0);

    gl_Position = MVP * vec4(vertex_position_modelspace, 1.0);
}
