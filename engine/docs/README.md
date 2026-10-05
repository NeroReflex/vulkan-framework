# Surfel global illumination

This book describes the renderer as it runs today. Read it in order the first time. After that, each chapter stands on its own.

The speed comes from chapter 4. Surfels are not walked in a tree. Each live surfel is linked into one cell of a spatial hash, and a pixel probes a few neighboring cells. The older Morton-code sort and Karras bounding-volume hierarchy are still compiled, but they are not dispatched. Chapter 10 says what those files are and why they must not be treated as the live index.

## Chapters

1. [The frame](01-the-frame.md) — what runs, in order, from the G-buffer to the swapchain.
2. [Memory](02-memory.md) — the surfel record, the stats block, the two halves, and the scratch buffer.
3. [Lifetime](03-lifetime.md) — mark, hole scan, commit, and when the index is allowed to change.
4. [Spatial hash](04-spatial-hash.md) — the index that queries actually use.
5. [Queries](05-queries.md) — point search, allocation search, and the same-frame staging list.
6. [Discovery](06-discovery.md) — which surfel covers each pixel.
7. [Spawn](07-spawn.md) — creating surfels and the direct-light image.
8. [Indirect light](08-indirect-lighting.md) — `surfel_rt`, the empty GI ray generator, and virtual point lights.
9. [Composite](09-composite.md) — final shading, the red surfel overlay, HDR, and present.
10. [Legacy tree](10-legacy-bvh.md) — shaders that still build, and the dispatch that no longer calls them.
11. [Changing it safely](11-maintenance.md) — invariants, atomics, and the mistakes that bring back popping or a stall.
12. [Moving surfels](12-moving-surfels.md) — node-local centers, the world-center pass, and skinned BLASes.
13. [Skinning](13-skinning.md) — tar skeleton layout, compute chain, and barriers.

## Source map

| Concern | Where |
|---|---|
| Frame order on the CPU | `engine/src/rendering/system.rs` |
| GI command buffer | `engine/src/rendering/pipeline/global_illumination.rs` |
| Limits shared with shaders | `engine/shaders/config.glsl` |
| Surfel record, queries, allocation | `engine/shaders/surfel.glsl` |
| Scratch layout, including the hash | `engine/shaders/surfel_reorder/build_layout.glsl` |
| Hash build | `engine/shaders/surfel_reorder/surfel_keys.comp` |
| Death, holes, commit | `surfel_mark.comp`, `surfel_prefix.comp`, `surfel_commit.comp`, `surfel_index_plan.comp` |

Shader binaries are embedded with `inline_spirv!` in `global_illumination.rs`. A comment of the form `lbvh v42 node surfels` next to those includes is only a rebuild tag. The name is historical. The running index is the hash in chapter 4.
