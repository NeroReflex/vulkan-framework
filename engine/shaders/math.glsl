#ifndef _MATH_
#define _MATH_ 1

#include "random.glsl"

bool is_below_horizon(vec3 surface_normal, vec3 in_normal) {
    return dot(surface_normal, normalize(in_normal)) < 0.0;
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

#endif // _MATH_
