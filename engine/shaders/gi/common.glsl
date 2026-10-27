#ifndef _GI_COMMON_
#define _GI_COMMON_

#include "../config.glsl"
#include "../compress.glsl"

#define PI 3.14159265358979
#define INVALID_ID 0xFFFFFFFFu

// Avalanching 32-bit integer hash (lowbias32). rust: hash_u32
uint hash_u32(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

uint hash_combine(uint a, uint b) {
    return hash_u32(a ^ (hash_u32(b) + 0x9e3779b9u + (a << 6u) + (a >> 2u)));
}

// PCG step; the high 24 bits are uniform in [0, 2^24).
uint rng_next(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rng_float(inout uint state) {
    return float(rng_next(state) >> 8u) * (1.0 / 16777216.0);
}

uint rng_seed(uint a, uint b, uint c) {
    return hash_combine(hash_combine(a, b), c);
}

float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// Orthonormal basis around n (Duff et al. 2017).
void make_basis(vec3 n, out vec3 t, out vec3 b) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float k = n.x * n.y * a;
    t = vec3(1.0 + s * n.x * n.x * a, s * k, -s * n.x);
    b = vec3(k, s + n.y * n.y * a, -n.y);
}

// Cosine-weighted direction around n, pdf = cos(theta) / pi.
vec3 sample_cosine(vec3 n, float u1, float u2) {
    float r = sqrt(u1);
    float phi = 2.0 * PI * u2;
    vec3 t, b;
    make_basis(n, t, b);
    float z = sqrt(max(0.0, 1.0 - u1));
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * z);
}

// Uniform point on the unit disk spanned by t and b, pdf = 1 / pi.
vec3 sample_disk(vec3 t, vec3 b, float u1, float u2) {
    float r = sqrt(u1);
    float phi = 2.0 * PI * u2;
    return t * (r * cos(phi)) + b * (r * sin(phi));
}

vec3 offset_origin(vec3 p, vec3 n) {
    return p + n * RAY_NORMAL_OFFSET;
}

// Partition-of-unity kernel of a surfel (center c, normal nc, radius r)
// evaluated at a receiver p with normal np. rust: surfel_kernel
float surfel_kernel(vec3 p, vec3 np, vec3 c, vec3 nc, float r) {
    vec3 d = p - c;
    float d2 = dot(d, d);
    float r2 = r * r;
    if (d2 >= r2) {
        return 0.0;
    }
    if (dot(np, nc) < SURFEL_KERNEL_MIN_COS) {
        return 0.0;
    }
    if (abs(dot(d, nc)) > SURFEL_KERNEL_PLANE_FRACTION * r) {
        return 0.0;
    }
    float t = 1.0 - d2 / r2;
    return t * t;
}

// Radius a surfel spawned at distance `dist` from the eye should have.
float target_radius(float dist) {
    return clamp(dist * gi_data.pixel_angle * SURFEL_SCREEN_RADIUS_PX, SURFEL_MIN_RADIUS, SURFEL_MAX_RADIUS);
}

#endif // _GI_COMMON_
