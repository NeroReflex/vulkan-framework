// Vulkan TransformMatrixKHR is three rows of four floats, translation in the
// last component of each row. GLSL mat4() takes columns, so this builds the
// matrix the raster path and the surfel node buffer both use.
mat4 row_major_3x4(vec4 r0, vec4 r1, vec4 r2) {
    return mat4(
        vec4(r0.x, r1.x, r2.x, 0.0),
        vec4(r0.y, r1.y, r2.y, 0.0),
        vec4(r0.z, r1.z, r2.z, 0.0),
        vec4(r0.w, r1.w, r2.w, 1.0)
    );
}
