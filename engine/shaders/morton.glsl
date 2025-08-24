#ifndef _MORTON_
#define _MORTON_ 1

// It is VERY important that this value is the maximum possible morton code
// because this way out-of-scale surfels will be sorted to the end of the array
// for easy removal in the sorting step.
#define MORTON_OUT_OF_SCALE 0xFFFFFFFFu

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
uint expandBits(uint v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
uint morton3D(in const vec3 v)
{
    const float x = min(max(v.x * 1024.0f, 0.0f), 1023.0f);
    const float y = min(max(v.y * 1024.0f, 0.0f), 1023.0f);
    const float z = min(max(v.z * 1024.0f, 0.0f), 1023.0f);
    const uint xx = expandBits(uint(x));
    const uint yy = expandBits(uint(y));
    const uint zz = expandBits(uint(z));
    return xx * 4 + yy * 2 + zz;
}

uint morton3D(in const vec3 eye_position, in const vec3 surfel_center, in const vec2 clip_space)
{
    const float range = abs(clip_space.y) - abs(clip_space.x);
    
    const vec3 min_allowed_position = eye_position - vec3(range);
    const vec3 max_allowed_position = eye_position + vec3(range);

    if (any(lessThan(surfel_center, min_allowed_position)) || any(greaterThan(surfel_center, max_allowed_position))) {
        return MORTON_OUT_OF_SCALE;
    }

    const vec3 center_converted = (surfel_center - min_allowed_position) / (max_allowed_position - min_allowed_position);
    //if (any(lessThan(center_converted, vec3(0.0))) || any(greaterThan(center_converted, vec3(1.0)))) {
    //    return MORTON_OUT_OF_SCALE;
    //}

    return morton3D(center_converted);
}

#endif // _MORTON_
