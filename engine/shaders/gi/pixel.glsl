#ifndef _GI_PIXEL_
#define _GI_PIXEL_

// Requires gbuffer.glsl and status.glsl.

ivec2 render_size() {
    return ivec2(gi_data.render_width, gi_data.render_height);
}

bool in_render_area(ivec2 px) {
    return all(greaterThanEqual(px, ivec2(0))) && all(lessThan(px, render_size()));
}

// Reads the primary hit of a pixel. The G-buffer normal is already on the
// visible side of the triangle (mesh_rendering.frag); it may point slightly away
// from the eye at smooth silhouettes and must not be flipped by the view.
bool load_pixel(ivec2 px, out vec3 pos, out vec3 n, out vec3 albedo, out uint inst) {
    pos = vec3(0.0);
    n = vec3(0.0, 1.0, 0.0);
    albedo = vec3(0.0);
    inst = 0xFFFFFFFFu;
    const uvec4 ids = texelFetch(gbuffer_instance_id, px, 0);
    if (ids.x == 0xFFFFFFFFu || ids.y == 0xFFFFFFFFu) {
        return false;
    }
    if (texelFetch(gbuffer_depth, px, 0).r >= 1.0) {
        return false;
    }
    pos = texelFetch(gbuffer[0], px, 0).xyz;
    vec3 raw_n = texelFetch(gbuffer[1], px, 0).xyz;
    float len = length(raw_n);
    if (len <= 0.0) {
        return false;
    }
    n = raw_n / len;
    albedo = texelFetch(gbuffer[2], px, 0).rgb;
    inst = ids.y;
    return true;
}

// Pixel of world position x in the previous frame.
bool reproject(vec3 x, out ivec2 q) {
    vec4 clip = gi_data.prev_proj * (gi_data.prev_view * vec4(x, 1.0));
    q = ivec2(-1);
    if (clip.w <= 0.0) {
        return false;
    }
    vec2 ndc = clip.xy / clip.w;
    vec2 pixel = (ndc * 0.5 + 0.5) * vec2(render_size());
    q = ivec2(floor(pixel));
    return in_render_area(q);
}

// Whether a sample stored for surface (xq, nq) may be reused at (x, n).
bool similar_surface(vec3 x, vec3 n, vec3 xq, vec3 nq) {
    if (dot(n, nq) < RESTIR_NORMAL_THRESHOLD) {
        return false;
    }
    float dist = distance(get_eye_position(), x);
    return abs(dot(xq - x, n)) <= RESTIR_PLANE_THRESHOLD * dist;
}

#endif // _GI_PIXEL_
