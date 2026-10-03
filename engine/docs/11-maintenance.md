# 11. Changing it safely

The usual way this pipeline breaks is a pass that assumes an invariant another pass no longer holds. These are the ones that are load-bearing.

## Invariants

1. **Committed slot ids are stable.** Discovery, the overlapping image, and surfel RT store them. Commit may write a slot only when that slot was `DEAD` and absent from the hash being queried, or when it is past the old `high_water` and was never in the hash.
2. **The hash describes `OFF_IDS_A` from the same rebuild.** Skipping the live-id prefix while still skipping the hash is correct. Rebuilding one and not the other is not. The plan flag is what keeps them together.
3. **Queries run after the hash barrier.** `barrier_bvh_publish` plus the compute barriers in `dispatch_keys` exist so heads, `next`, and `ids` are visible. An atomic in the insert shader does not replace that barrier for other dispatches.
4. **Empty heads are `0xFFFFFFFF`, not 0.** Live-list slot 0 is real.
5. **Staging slots are unpublished until `READY`.** Scans must use the acquire load on flags. Do not read `position` from a slot that failed that test.
6. **Locks are one try.** A loop on `lock_surfel` in a full-screen ray generator will hang the device if two invocations in a subgroup need locks in opposite orders. The `POPULATE_VISIBLE_SURFELS_LIST` path still spins. It is compiled out. Do not enable it without removing the spin.
7. **`inline_spirv!` embeds the shader at Rust compile time.** Editing a `.comp` file does nothing until `cargo build` rebuilds `artrtic`. The debugger launches the debug profile, not `--release`.

## Atomics, and what they cost

| Atomic | Where | How often |
|---|---|---|
| `atomicExchange` on a hash head | hash insert | once per live surfel, only on a dirty frame |
| compare-exchange on `unordered_surfels` | allocation | once per successful reservation, at most 256 per frame |
| compare-exchange on `flags` | `lock_surfel` | once per pixel that has an overlapping surfel, inside surfel RT |
| `atomicOr` of `DEAD` | mark | once per death |
| `atomicStore` of `index_topology_dirty` | mark and commit | once per death or per commit that placed something |

The lock in surfel RT is the one that scales with pixels, not with surfels. A huge surfel covers a huge number of pixels, and every one of those pixels attempts the same flag word. They fail fast, but the attempts are still atomics on one address. If a later change needs per-pixel work on that surfel, do not take this lock per pixel. Take it once, or do the work from a compacted list of ids.

Hash insert atomics are spread across buckets. They are not the open-scene cost.

## Radius edits and the hash

Mark may store a new radius without setting `index_topology_dirty`. The surfel stays in the level chosen at the last insert. Queries probe every level, but only a 3×3×3 at that level's cell size. If the radius grows across a level boundary (32 or 64) and the hash is not rebuilt, a point can lie inside the new sphere and outside the 3×3×3 of the old fine cell.

If you need radius updates to be exact, set `index_topology_dirty` in that branch of mark. The following frame pays for a hash rebuild, which is cheap next to a full-screen probe.

## Far plane

`is_out_of_range` is `distance(eye, center) > abs(far)`. It used to be an axis-aligned cube centered on the eye with half-extent `far - near`. That cube moved with the camera and killed surfels that were still in the world. Do not restore the cube.

`radius_from_camera_distance` still uses near and far as the endpoints of the 10..120 map. A wrong near or far in `reconstructNearFarFromCamera` changes every radius and makes mark rewrite them.

## Coverage after frame 1

Spawn allocates only while `gi_reuse_frames < 2`. Surfel RT allocates only while `gi_reuse_frames == 0`. After that, new screen area is covered only by surfels that already exist and still pass the far-plane test. If you want the courtyard to fill in as the camera arrives, allocation has to run on uncovered pixels every frame, with the same 256-per-frame cap.

## Overlapping image

Clear to `0xFFFFFFFF` happens every frame. Discovery is the only store. Anything that reads the image at a pixel discovery did not write will see "no surfel". That includes a stride greater than 1, an early return, and a hash miss.

## Adding a pass

Put it after `barrier_bvh_publish` if it reads the hash, and before mark of the next frame if it wants to clear `latest_contribution`. If it writes surfel payloads, take `lock_surfel` or be the sole writer of those fields (mark is the sole writer of `DEAD` and of the radius update, and it runs before any query).

Do not write `OFF_IDS_A`, hash heads, or `HDR_KEY_COUNT` from a pixel shader. Those words are the index. The plan, the prefix, and `surfel_keys.comp` own them.

## Files that look live and are not

Chapter 10. Deleting them is safe only after you also delete the Rust pipelines, the push-constant ranges, and the `BVHNode` binding if nothing else declares it. Leaving them compiled is harmless as long as they stay off the command buffer.
