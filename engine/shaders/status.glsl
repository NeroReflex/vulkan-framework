#ifndef _STATUS_
#define _STATUS_

#include "morton.glsl"

#ifndef STATUS_DESCRIPTOR_SET
    #define STATUS_DESCRIPTOR_SET 2
#endif

layout(std140, set = STATUS_DESCRIPTOR_SET, binding = 0) uniform camera_uniform {
	mat4 viewMatrix;
	mat4 projectionMatrix;
} camera;

layout(std430, set = STATUS_DESCRIPTOR_SET, binding = 1) readonly buffer directional_lights
{
    light_t light[];
};

vec3 get_eye_position() {
    return vec3(camera.viewMatrix[3][0], camera.viewMatrix[3][1], camera.viewMatrix[3][2]);
}

vec2 reconstructNearFarFromCamera()
{
    // GLSL mat4 is column-major: projectionMatrix[c][r]
    // A = m22 = proj[2][2], B = m32 = proj[3][2] in 0-based row/col notation
    const float A = camera.projectionMatrix[2][2];
    const float B = camera.projectionMatrix[3][2];

    // Detect D3D (NDC z in [0,1]) vs GL (NDC z in [-1,1]) by checking typical pattern:
    // For GL: proj[2][3] == -1.0 and formula uses (A,B) as below.
    // For D3D: proj[2][3] == 1.0 (or proj[2][3] sign differs). We'll branch accordingly.
    const float signW = camera.projectionMatrix[2][3]; // typically -1.0 (GL) or 1.0 (D3D)
    if (signW < 0.0) {
        // OpenGL style (NDC z in [-1,1])
        // A = (zFar+zNear)/(zNear-zFar)
        // B = (2*zFar*zNear)/(zNear-zFar)
        // Solve:
        float zNear = B / (A - 1.0);
        float zFar  = B / (A + 1.0);

        // handle infinite far: when zFar is extremely large or B == 0
        if (abs(B) < 1e-12) {
            // infinite far plane case: typical A = -1, B = 0 -> derive near from A if possible
            // For infinite far, projection often sets m22 = -1, m32 = -2*zNear (but forms vary).
            // Fallback: estimate zNear from proj[2][2] if possible, otherwise set defaults.
            // Here set zFar large and compute zNear using alternative relation if available.
            zFar = 1e30;
        }

        return vec2(zNear, zFar);
    } else {
        // D3D/Vulkan style (NDC z in [0,1])
        // A = zFar/(zNear-zFar)
        // B = (zFar*zNear)/(zNear-zFar)
        // Solve:
        float zNear = B / A;
        float zFar  = B / (A - 1.0);

        if (abs(A) < 1e-12) {
            zFar = 1e30;
        }

        return vec2(zNear, zFar);
    }
}

#endif // _STATUS_