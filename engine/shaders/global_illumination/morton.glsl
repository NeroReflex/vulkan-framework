#ifndef _MORTON_
#define _MORTON_ 1

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

#endif // _MORTON_
