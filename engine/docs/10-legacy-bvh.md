# 10. Legacy tree

These files still compile, because `global_illumination.rs` still builds pipelines for them. **Nothing in `record_rendering_commands` binds those pipelines.**

| File | What it did |
|---|---|
| `surfel_key_radix.comp` | Four-pass radix sort of Morton keys. |
| `karras.glsl` | Karras 2012 range and split, used to build a binary radix tree from sorted keys. |
| `surfel_lbvh.comp` | Wrote `BVHNode.left`, `right`, and `parent`, and leaf-parent ids. |
| `bvh_aabb.comp` | Leaf-up AABB refit using `tree[node].flags` as an atomic arrival counter. |
| `morton.glsl` | 30-bit Morton codes of centers quantized into a scene AABB. |

`surfel_keys.comp` used to emit those Morton keys. It is now the hash build. The Rust function is still named `dispatch_keys`.

`struct BVHNode` and binding 2 still exist so those shaders, and any reader that declares the binding, keep a valid layout. Queries do not load `tree[]`.

## Why it was removed from the frame

The tree was correct once the radix scan was split across dispatches (a single dispatch cannot read another workgroup's histogram). It was still the wrong query for this data.

Surfels are spheres with a large radius relative to their spacing. Parent boxes contain a lot of empty space. An open view made almost every child test succeed, so each pixel walked the cap (`LBVH_WALK_LIMIT`, 64) and pulled a 64-byte node each step from an unpredictable address. Indoor views culled early and looked fine. The courtyard did not.

The hash pays a fixed neighborhood instead of a data-dependent walk. Build cost is one atomic per live surfel.

## If you turn the tree back on

Do not point queries at `tree[]` unless you also dispatch `surfel_lbvh` and `bvh_aabb` again, after a real sort. A stale or zeroed tree returns misses and looks like "surfels popped out". The hash heads live in the same scratch region as the old keys (`OFF_HASH_HEAD = OFF_KEYS_A`). Rebuilding keys in that region destroys the hash. One index at a time.

The radix shader's histogram, scan, and scatter are three phases on purpose. Do not fold them back into one dispatch. Workgroup barriers do not publish writes to other workgroups.

## Names you will trip over

| Name in the source | What it is now |
|---|---|
| `bvh_search` | Hash point probe. |
| `linear_search_ordered_surfel_for_allocation` | Hash probe with instance and too-close tests. |
| `lbvh_empty` | `live_count <= 0`. |
| `lbvh_key_count` | `live_count`. |
| `dispatch_keys` | Hash clear and insert. |
| `HDR_KEY_COUNT` | Live count at the last hash build. |
| Comment `lbvh v42 node surfels` | Cache-bust for `inline_spirv!`. Not a version of the tree. |
