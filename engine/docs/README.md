# Surfel global illumination

This book describes the renderer as it runs today. Read it in order the first time. After that, each chapter stands on its own.

Indirect light comes from a cache of **surfels**: small disks laid on the scene's surfaces. Each surfel estimates the light arriving at its patch of surface and re-emits it as a Lambertian **disk light**. Every pixel then picks one surfel light per frame with **ReSTIR**: resampled importance sampling, temporal and spatial reuse, and one visibility ray. The result is accumulated over a few frames.

Nothing in this pipeline sorts or moves surfels. A surfel keeps its slot from birth to eviction. Lookups go through a multi-level hashed grid that is rebuilt from scratch every frame in about 0.07 ms, and every query reads a fixed number of buckets.

## Chapters

1. [The frame](01-the-frame.md) — what runs, in order, from the G-buffer to the swapchain, and which counters invalidate history.
2. [Memory](02-memory.md) — the surfel record, the state words, light records, reservoirs, images, and the descriptor bindings.
3. [Lifetime](03-lifetime.md) — the free stack, the alive lists, maintenance, eviction, and slot generations.
4. [Spatial hash](04-spatial-hash.md) — the 8-level hashed grid and why a lookup is O(1).
5. [Queries](05-queries.md) — the surfel kernel, the partition of unity, and the three lookups built on it.
6. [Coverage and spawn](06-coverage-and-spawn.md) — how the screen gets covered, and how over-coverage is pruned.
7. [Surfel update](07-surfel-update.md) — direct light, area, indirect light, and offscreen surfels.
8. [Surfel lights](08-surfel-lights.md) — radiance, the disk geometry term, the integer flux CDF, and light tiles.
9. [ReSTIR](09-restir.md) — initial candidates, temporal and spatial reuse, Z-normalization, shading, and accumulation.
10. [Composite](10-composite.md) — final shading, debug views, HDR, UI, and present.
11. [Testing](11-testing.md) — the path-traced reference, captures, the compare tool, the self-test, timings, and measured results.
12. [Changing it safely](12-maintenance.md) — invariants, the mistakes that were already made once, and shader build quirks.
13. [Moving surfels](13-moving-surfels.md) — instances, generations, the scene file, and the preview.
14. [Skinning](14-skinning.md) — skeleton layout, the compute chain, and its barriers.

## Source map

| Concern | Where |
|---|---|
| Frame order on the CPU | `engine/src/rendering/system.rs` |
| Status block, history, captures, self-test driver | `engine/src/rendering/gi_driver.rs` |
| GI command buffer, buffers, barriers | `engine/src/rendering/pipeline/global_illumination.rs` |
| CPU mirrors of the shader math and constants, unit tests | `engine/src/rendering/pipeline/gi_math.rs` |
| Limits shared with shaders | `engine/shaders/config.glsl` |
| Surfel record, bindings, grid, lights | `engine/shaders/gi/surfels.glsl` |
| Every GI pass | `engine/shaders/gi/*.comp`, `*.rgen` |
| Final shading and debug views | `engine/shaders/final_rendering/final_rendering.frag` |
| Self-test, capture writer, compare tool | `engine/src/rendering/gi_selftest.rs`, `gi_capture.rs`, `engine/src/bin/artrtic-compare.rs` |

## Quick checks

```bash
engine/shaders/gi/validate.sh                       # every GI shader compiles and passes spirv-val
cargo test --lib gi_math                            # CPU math tests
env -u LD_PRELOAD ART_RTIC_NO_UI=1 ART_RTIC_MAX_FRAMES=620 \
    ART_RTIC_CAPTURE=/tmp/gi/cap@600 ART_RTIC_SELFTEST=600 cargo run --bin artrtic
cargo run --bin artrtic-compare -- /tmp/gi/cap f00600 --max-bias 5 --max-block-rmse 25
```

`env -u LD_PRELOAD` is needed on the development laptop, where `libnvidia-hide.so` hides the discrete GPU. Chapter 11 explains every switch.
