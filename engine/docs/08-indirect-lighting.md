# 8. Indirect light

Three launches follow spawn. Only one of them currently changes surfel irradiance.

## Surfel RT

`surfel_rt.rgen`, full viewport. `POPULATE_VISIBLE_SURFELS_LIST` is 0, so each thread reads `surfelOverlappingImage` at its pixel. If the id is missing or `lock_surfel` fails, the thread returns. Many pixels share one surfel. The first lock wins and does the work. The others leave. That is intentional. The lock is not retried, because a retry loop inside a ray-generation shader can deadlock a subgroup.

The winner runs until `frame_contributions` reaches `FIRST_PASS_RAYS` (16). Each primary ray:

1. Picks a random direction in the hemisphere of the surfel normal (`random_ray_above_horizon`).
2. Traces it against the mesh acceleration structure, from the surfel center, with `tMin = CLOSEST_INTERSECTION_DISTANCE` (0.4) so the ray does not hit the surface it starts on.
3. On a miss, the bounce ends.
4. On a hit, it looks for a surfel at the hit point.
   - If `gi_reuse_frames == 0`, it may allocate one (`find_surfel_or_allocate_new` with `allocate_locked = true`).
   - Otherwise it only calls `bvh_search` (the hash point search). No new surfels on later frames.
5. A brand-new bounce surfel gets directional light from `rt_directional_light` and is kept locked.
6. An existing surfel must be locked or the bounce stops.
7. The hit surfel's `latest_contribution` is cleared.

The loop allows `MAX_RAY_DEPTH` (2) bounces. Lighting is then folded backward: each emitter's `projected_irradiance` is added onto the previous surfel with `addIndirectLightContribution`. Locks are released in reverse order. The primary surfel's `frame_contributions` increments when at least one bounce existed, and its `latest_contribution` is cleared before unlock.

`projected_irradiance` treats the emitter as a point light: averaged irradiance plus direct light, times the emitter albedo, times the clamped cosine on the receiver. There is no visibility test between the two surfels. The ray trace already decided they face each other along this path.

`addIndirectLightContribution` saturates around 16384 samples by leaking a fraction of the accumulated irradiance, so a surfel that lives a long time does not overflow and can forget stale light.

Once `frame_contributions` is 16, later frames still launch the ray generator and still take the lock, but the ray loop does not run. The thread clears `latest_contribution`, unlocks, and returns. The cost after the first frames is a full-screen image load and a contended atomic on the surfel's flag word, not 16 rays per pixel.

## GI ray generator

`global_illumination.rgen` is launched at full resolution after surfel RT. The surfel path is:

```glsl
#if ENABLE_SURFELS
#else
    // monte carlo path that writes outputImage[0]
#endif
```

`ENABLE_SURFELS` is 1, so this shader returns after reading the G-buffer and the reuse counter. It does not write the GI image. The `#else` branch is the non-surfel path tracer: 8 primary rays, 2 bounces, a shadow ray per directional light, blended into `outputImage[0]` by `gi_reuse_frames`.

Do not "clean up" the empty branch by enabling that path while surfels are on. It would trace a full-screen path on top of surfel RT.

The miss and closest-hit shaders for this pipeline are still bound. They are unused while the ray generator never calls `traceRayEXT`.

## Virtual point lights

`surfel_vpl.comp` runs at `SURFEL_VPL_QUERY_STRIDE` 2, so about a quarter of the pixels. Groups are 32×16. Sky pixels return immediately.

For each remaining pixel it loads the G-buffer position and calls `bvh_search`. On a hit it evaluates `projected_irradiance` and then does not store it. The comment in the shader says the add into `outputImage` is still to do.

So the pass is a quarter-resolution hash probe with a dead result. It is cheap compared with the old tree, and it is not part of the picture you see. `VIRTUAL_POINT_LIGHTS_PER_PIXEL` (4) is unused here. The loop samples one surfel.

## Where the picture's light comes from

| Buffer | Writer | When |
|---|---|---|
| `outputImage[1]` direct | spawn | frame 0 only, then reused |
| `outputImage[0]` indirect | GI ray generator's non-surfel branch | never, while surfels are enabled |
| surfel `irradiance_*` and `direct_light_*` | spawn and surfel RT | as described above |
| red overlay | final shading, from the overlapping id | every frame, if `SHOW_SURFELS` is 1 |

If the goal of a change is "surfels light the scene", the missing store is either "scatter surfel irradiance into `outputImage[0]` from the overlapping id" or "finish the VPL write". The hash does not do that by itself.
