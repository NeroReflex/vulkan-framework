#ifndef _AABB_
#define _AABB_

struct AABB
{
    vec3 vMin;
    vec3 vMax;
};

bool vertexContainedInAABB(in const AABB aabb, in const vec3 vtx) {
    return
        ((aabb.vMin.x <= vtx.x) && (aabb.vMax.x >= vtx.x)) &&
        ((aabb.vMin.y <= vtx.y) && (aabb.vMax.y >= vtx.y)) &&
        ((aabb.vMin.z <= vtx.z) && (aabb.vMax.z >= vtx.z));
}

bool triangleContainedInAABB(in const AABB aabb, in const vec3 v1, in const vec3 v2, in const vec3 v3) {
    return vertexContainedInAABB(aabb, v1) && vertexContainedInAABB(aabb, v2) && vertexContainedInAABB(aabb, v3);
}

AABB expandAABB(in const AABB aabb, in const vec3 v) {
	AABB result = aabb;

	result.vMin.x = min(result.vMin.x, v.x);
	result.vMin.y = min(result.vMin.y, v.y);
	result.vMin.z = min(result.vMin.z, v.z);

	result.vMax.x = min(result.vMax.x, v.x);
	result.vMax.y = min(result.vMax.y, v.y);
	result.vMax.z = min(result.vMax.z, v.z);

	return result;
}

AABB combineAABBs(in const AABB first, in const AABB second)
{
    AABB result;
    result.vMin.x = min(first.vMin.x, second.vMin.x);
    result.vMin.y = min(first.vMin.y, second.vMin.y);
    result.vMin.z = min(first.vMin.z, second.vMin.z);
    result.vMax.x = max(first.vMax.x, second.vMax.x);
    result.vMax.y = max(first.vMax.y, second.vMax.y);
    result.vMax.z = max(first.vMax.z, second.vMax.z);
    return result;
}

AABB compatAABB(in const vec3 vMin, in const vec3 vMax)
{
    AABB result;
    result.vMin = vMin;
    result.vMax = vMax;
    return result;
}

AABB compatAABB(in const mat4 t, in const vec3 vMin, in const vec3 vMax) {
	vec4 v[8];
	v[0] = vec4(vMin.x, vMin.y, vMin.z, 1.0) * t; // 000
	v[1] = vec4(vMin.x, vMin.y, vMax.z, 1.0) * t; // 001
	v[2] = vec4(vMin.x, vMax.y, vMin.z, 1.0) * t; // 010
	v[3] = vec4(vMin.x, vMax.y, vMax.z, 1.0) * t; // 011
	v[4] = vec4(vMax.x, vMin.y, vMin.z, 1.0) * t; // 100
	v[5] = vec4(vMax.x, vMin.y, vMax.z, 1.0) * t; // 101
	v[6] = vec4(vMax.x, vMax.y, vMin.z, 1.0) * t; // 110
	v[7] = vec4(vMax.x, vMax.y, vMax.z, 1.0) * t; // 111
	

    for (uint i = 0; i < 8; ++i) {
		v[i] = vec4( v[i].xyz / v[i].w, 1.0);
    }

    AABB result;
	result.vMin = vec3(
        min(v[0].x, min(v[1].x, min(v[2].x, min(v[3].x, min(v[4].x, min(v[5].x, min(v[6].x, v[7].x))))))),
        min(v[0].y, min(v[1].y, min(v[2].y, min(v[3].y, min(v[4].y, min(v[5].y, min(v[6].y, v[7].y))))))),
        min(v[0].z, min(v[1].z, min(v[2].z, min(v[3].z, min(v[4].z, min(v[5].z, min(v[6].z, v[7].x)))))))
    );

	result.vMax = vec3(
        max(v[0].x, max(v[1].x, max(v[2].x, max(v[3].x, max(v[4].x, max(v[5].x, max(v[6].x, v[7].x))))))),
        max(v[0].y, max(v[1].y, max(v[2].y, max(v[3].y, max(v[4].y, max(v[5].y, max(v[6].y, v[7].y))))))),
        max(v[0].z, max(v[1].z, max(v[2].z, max(v[3].z, max(v[4].z, max(v[5].z, max(v[6].z, v[7].x)))))))
    );

    return result;
}

#endif // _AABB_
