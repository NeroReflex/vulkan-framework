# 12. Changing it safely

This pipeline usually breaks when one pass assumes something another pass no longer guarantees. These are the guarantees the passes rely on, and the traps that have already been hit once.

## Invariants

1. **Slot ids are stable, and payloads never move.** The overlay, reservoirs, light tiles, and grid all store ids. Only maintain frees slots. Only the surfel update and coverage allocate. Any new way of allocating or freeing must also show up in `ST_FREE`, `ST_EVICTED`, or `ST_DOOMED`. Otherwise `ST_STABLE_FRAMES` keeps counting while the cache changes (chapter 9).
2. **No pass both pushes and pops the free stack.** That is what makes `surfel_alloc()` safe without a lock (chapter 3). A pass that needs to free and allocate must be split in two with a barrier between.
3. **The grid describes exactly the survivors.** Grid phase 0 freezes `ST_SURVIVORS`, and the surfel update launch is limited to it. Newborns are appended after the survivors and are found by queries only from the next frame. Never index the grid with `ST_ALIVE_OUT`.
4. **Every allocation bumps `generation`, and every free clears the light record.** Reservoirs rely on both to detect a recycled slot. Pool reset keeps generations.
5. **`SURFEL_MAX_RADIUS ≤ GRID_BASE_CELL · 2^(GRID_LEVELS-1) / 2`.** Otherwise a surfel spans more than 2 × 2 × 2 cells, and `surfel_buckets` silently drops memberships.
6. **`MAX_SURFELS × FLUX_QUANT_MAX < 2^32`.** The flux CDF is a `uint`. Raising the pool size means lowering the quantization.
7. **The ReSTIR target and its support test must agree exactly.** `Z` uses `record_supports`, and the shaded value uses `disk_geometry`. If you change the geometry term, keep `G ≥ DISK_SUPPORT_FLOOR` wherever `disk_supports` is true and `G = 0` elsewhere, and keep `disk_support_matches_geometry` passing.
8. **The pmf in a tile entry is the exact selection probability.** If you change the draw, change the pmf with it. A pmf that is merely close biases every pixel.
9. **Constants marked `rust:` in `config.glsl` and `surfels.glsl`** are mirrored in `gi_math.rs` and must match: strides (`SURFEL_STRIDE` 128, `RESERVOIR_STRIDE`, `TILE_ENTRY_STRIDE`, `LIGHT_RECORD_STRIDE` 32), `ST_*` words, grid constants, and hash functions. The self-test compares GPU data with CPU mirrors, so a mismatch shows up there first.

## Shader traps

- **`inline_spirv!` does not track includes.** Editing a `.glsl` header changes nothing until `global_illumination.rs` is rebuilt. Run `touch engine/src/rendering/pipeline/global_illumination.rs` before `cargo build`.
- **glslang emits invalid SPIR-V for a struct-valued ternary** that mixes a buffer element and a local struct (`c ? buf[i] : local`). The driver then rejects the module with `VUID-VkShaderModuleCreateInfo-pCode-08737` (`OpStore` type mismatch). Use `if`/`else`. `validate.sh` catches this.
- **`payload` is overwritten by every trace,** including the shadow rays inside `direct_irradiance`. Copy what you need into locals right after `trace_closest`.
- **Image arrays take one binding slot per element** in the framework's descriptor writer. A new binding goes after 27, not after 20.

## Things that look reasonable and are wrong

- **Spreading a surfel's power over the disk when the surface covers only part of it.** That is why area and direct light count only on-surface samples (chapter 7). Without it, surfels on ledges over-emit by about 30%.
- **Point-light falloff \(\cos\cos/(d^2 + r^2)\).** It is off by up to 4× near the disk, and near the disk is where most of the indirect light comes from in interiors. Use `disk_geometry`.
- **Dividing by `M` instead of `Z`.** It is biased dark at every concave corner.
- **Letting offscreen surfels spawn more offscreen surfels.** The pool fills with surfaces nobody sees. Only on-screen surfels spawn by ray, and ray spawns stop under pool pressure.
- **Truncating at an uncovered hit.** Each uncovered hit loses its indirect light. Path continuation (chapter 7) keeps multibounce within a few percent.
- **Shrinking the emitting disk to \(\sqrt{A/\pi}\).** Tried, and it raises the error. The surfel center is not the centroid of the covered surface, so a smaller disk concentrates the power in the wrong place.
- **Resetting the history on camera moves.** Everything is reprojected or in world space. A reset also empties the surfel pool.
- **Converging a still image with a longer accumulation window alone.** The cache under it keeps changing and the reused samples are correlated, so the image keeps drifting by several percent per pixel. It converged only after the cache stopped changing and temporal reuse stopped (chapter 9).
- **Switching the surfel estimates to running means as soon as the camera stops.** Surfels spawned during the settle phase carry noisy first samples, and a running mean keeps them for thousands of frames. Wait until the surfel set has stopped changing.
- **Freezing offscreen spawns the moment the camera stops.** The pool is still filling up then. Freezing it at that point lost 5% of the indirect light, because missing offscreen surfels are missing lights.
- **Turning the G-buffer normal toward the eye.** At silhouettes this flips smooth normals into the surface, and those pixels go black. Orient the normal by the rasterized face instead (chapter 1).
- **A sampler with `max_lod = 0`.** Only mip 0 is ever read, which shows up as moiré on fine textures such as the curtains.

## Adding a pass

Put it between the barriers of the existing sequence in `record_rendering_commands`. Add a timestamp label if it costs anything. Use the compute layout from `gi/compute.glsl`, or the ray tracing set layout used by `surfel_update.rgen`. If it reads the grid, it must run after grid scatter. If it allocates, it must not also free. If it writes surfel fields that another pass reads in the same frame, make sure that other pass runs after the barrier. Then add the pass to `validate.sh`, and, if it changes data the self-test reads, add a check.
