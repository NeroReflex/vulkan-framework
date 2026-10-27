#ifndef _GI_RESERVOIR_
#define _GI_RESERVOIR_

// Requires surfels.glsl and pixel.glsl.

// rust: RESERVOIR_STRIDE = 32 bytes
struct Reservoir {
    // selected surfel (INVALID_ID when empty) and its slot generation
    uint id;
    uint gen;
    // unbiased contribution weight
    float W;
    // number of candidates the reservoir represents (0 = no surface)
    float M;
    // surface the reservoir was built for, used to validate reuse
    vec3 pos;
    uint nrm;
};

layout(set = SURFEL_SET, binding = 14, std430) buffer Reservoirs { Reservoir reservoirs[]; };

#define RES_TEMP 2u

uint res_current() {
    return gi_data.frame_index & 1u;
}

uint res_previous() {
    return (gi_data.frame_index + 1u) & 1u;
}

uint res_index(uint region, ivec2 px) {
    return region * gi_data.render_width * gi_data.render_height + uint(px.y) * gi_data.render_width + uint(px.x);
}

Reservoir empty_reservoir(vec3 pos, vec3 n, float M) {
    Reservoir r;
    r.id = INVALID_ID;
    r.gen = 0u;
    r.W = 0.0;
    r.M = M;
    r.pos = pos;
    r.nrm = compress_unit_vec(n);
    return r;
}

// The light record of the reservoir's surfel if that surfel still exists
// (same slot generation) and is a light; an empty record otherwise.
// Evicted slots have their record cleared, so no alive check is needed.
LightRecord reservoir_light(Reservoir r) {
    if (r.id >= MAX_SURFELS) {
        return empty_light_record();
    }
    LightRecord rec = light_rec[r.id];
    if (rec.info.z != r.gen) {
        return empty_light_record();
    }
    return rec;
}

bool reservoir_sample_valid(Reservoir r) {
    return uintBitsToFloat(reservoir_light(r).info.y) > 0.0;
}

#endif // _GI_RESERVOIR_
