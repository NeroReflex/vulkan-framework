#ifndef _MATH_
#define _MATH_ 1

#include "random.glsl"

#define PI 3.14159265f
#define HALF_PI 1.57079f
#define EPSILON 0.000000000000000000000000000001f
#define SQRT_3 1.73205080f
#define POSITIVE_INF (1.0f / 0.0f)
#define NEGATIVE_INF (-1.0f / 0.0f)
#define MIN_FLOAT -340282346638528859811704183484516925440.0000000000000000f
#define MAX_FLOAT 340282346638528859811704183484516925440.0000000000000000f
#define kEpsilon 0.00000001f

vec3 random_point_on_unit_sphere(inout uint state)
{
  float z = rnd(state) * 2.0 - 1.0;
  float t = rnd(state) * 6.28318530718; // 2 * PI
  float r = sqrt(1.0 - z * z);
  return vec3(r * cos(t), r * sin(t), z);
}

// Returns true if the out_normal is below the horizon defined by surface_normal
//
// This function normalizes out_normal, and out_normal MUST NOT be zero
// and be a direction pointing OUTWARD from the surface
bool is_below_horizon(vec3 surface_normal, vec3 out_normal) {
    return dot(surface_normal, normalize(out_normal)) < 0.0;
}

// Given a surface point and  its normal, finds if the other_point is below the horizon
// defined by the surface normal.
bool is_point_below_horizon(vec3 surface_point, vec3 surface_normal, vec3 other_point) {
    // here calculate a vector going from surface_point to other_point,
    // this way we will have an OUTGOING direction from the surface,
    // and we can then use the is_below_horizon function
    vec3 out_normal = other_point - surface_point;

    return is_below_horizon(surface_normal, out_normal);
}

vec3 random_ray_above_horizon(vec3 surface_normal, inout uint seed) {
    const vec3 random_point_on_sphere = random_point_on_unit_sphere(seed);
    vec3 ray_dir = reflect(surface_normal, random_point_on_sphere);
    return is_below_horizon(surface_normal, random_point_on_sphere) ? -ray_dir : ray_dir;
}

// Function to generate a biased random number
float biasedRandom(inout uint seed, uint count, float p) {
    // Generate a uniform random number in the range [0, 1)
    float uniformRandom = float(lcg(seed)) / float(1u << 24);
    // Apply the bias using the power function
    return float(count) * pow(uniformRandom, p);
}

// Function to calculate the relative probability of hitting a specific point
float relativeProbability(float x, uint count, float p) {
    return (p * pow(float(x) / float(count), p - 1.0)) / float(count);
}

float map(float value, float inMin, float inMax, float outMin, float outMax) {
    const float scale = (value - inMin) / (inMax - inMin);
    return outMin + scale * (outMax - outMin);
}

#endif // _MATH_
