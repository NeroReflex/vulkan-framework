#ifndef _MATH_
#define _MATH_ 1

bool is_below_horizon(vec3 surface_normal, vec3 in_normal) {
    return dot(surface_normal, normalize(in_normal)) < 0.0;
}

#endif // _MATH_
