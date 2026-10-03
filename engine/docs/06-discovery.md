# 6. Discovery

`surfel_discovery.comp` answers one question per pixel: which committed surfel covers the G-buffer position, if any?

It runs after the hash publish barrier and before spawn. Newborns from this frame are not visible to it. They were not committed yet. The overlapping image therefore lags spawn by one frame, which is the same lag as the hash.

## Dispatch

Rust divides the viewport by `SURFEL_DISCOVERY_QUERY_STRIDE` (1) and launches groups of 32×16. Stride 1 is required. The overlapping image is cleared to `0xFFFFFFFF` every frame, and `surfel_rt` plus final shading read it at full resolution. A stride of 2 leaves three quarters of the pixels empty and looks like missing surfels.

The shader multiplies `gl_GlobalInvocationID.xy` by the stride to get the texel. With stride 1 that is the pixel itself.

## Body

```text
if the texel is outside the image, or the mesh id is 0xFFFFFFFF:
    enabled = false
else:
    origin = gbuffer position
    instance_id = gbuffer instance
    radius = radius_from_camera_distance(eye, clip planes, origin)
    found = linear_search_ordered_surfel_for_allocation(origin, instance_id, radius)

if found is a real surfel id and the pixel is enabled:
    surfels[found].latest_contribution = 0
    imageStore(overlapping, pixel, found)
```

The reset of `latest_contribution` is what keeps a visible surfel alive. Mark increments that counter at the start of the next frame. Spawn used to be the only pass that cleared it, and spawn stops touching covered pixels after frame 2. Without this store, every surfel would age out after 120 frames even while it was on screen.

The store is skipped when the search returns miss, too-close, or busy. The pixel stays `0xFFFFFFFF` from the clear. Final shading draws no red there. Spawn, on later frames, treats that as "try to find or create a surfel".

## What this pass does not do

`POPULATE_VISIBLE_SURFELS_LIST` is 0. The workgroup compaction into the `discovered[]` buffer is compiled out. Surfel RT does not iterate that list. It reads the overlapping image per pixel instead.

Discovery does not allocate, lock, or trace rays. It is a full-screen hash probe plus one image store on hits. That is the dominant surfel cost when the camera faces a lot of geometry, and it is still far smaller than a divergent tree walk of fat bounds.

## Overlapping image contract

| Value | Meaning |
|---|---|
| `0xFFFFFFFF` | Clear, or discovery found nothing. |
| any other id | A committed slot. The pixel's world position was inside that sphere, on the same instance, at the moment of discovery. |

The id can become `DEAD` later in the frame only if mark ran before discovery, which it did, on the previous frame's set. Discovery does not observe deaths from the future. It can observe a surfel that spawn is about to cover with a better newborn. The newborn is not in the hash yet, so discovery keeps the old id. That is correct: the old sphere still contains the point.
