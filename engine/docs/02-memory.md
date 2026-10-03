# 2. Memory

All surfel state is a handful of device buffers created in `GILighting::new`. Shaders see them through one descriptor set (`output_descriptor_set`). Binding numbers below are that set.

## The surfel record

`struct Surfel` in `surfel.glsl` is 23 scalar fields, 92 bytes, matching `SURFEL_SIZE` in Rust (`23 * 4`).

| Field | Role |
|---|---|
| `instance_id` | Draw instance that owns the surface. A hit on a different instance is not a match. The same id indexes the node-matrix buffer. |
| `position_*` | World-space center. `surfel_world.comp` rewrites it from `bind_*` before mark and again before the hash. |
| `bind_*` | Center in the instance's local space at spawn (`inverse(node) * world hit`). |
| `bone` | `0xFFFFFFFF` for a rigid instance. Otherwise the palette joint whose matrix writes the world center. |
| `radius` | World-space sphere radius. Mark may rewrite it when the camera distance changes. |
| `normal` | Octahedral unit normal (`compress.glsl`). |
| `diffuse_*` | Albedo copied from the G-buffer at spawn. |
| `irradiance_*` | Sum of indirect samples. Divide by `contributions`. |
| `direct_light_*` | Sum of unoccluded directional light at spawn. |
| `contributions` | How many indirect samples have been added. |
| `frame_contributions` | How many primary rays `surfel_rt` has finished for this surfel. Caps at 16. |
| `flags` | See below. |
| `latest_contribution` | Frames since discovery or spawn last touched this surfel. |

Flags, from `surfel.glsl`:

| Bit | Name | Meaning |
|---|---|---|
| 0 | `SURFEL_FLAG_LOCKED` | One invocation holds the record for a read-modify-write. |
| 1 | `SURFEL_FLAG_PRIMARY` | Reserved. Not required by the hash. |
| 2 | `SURFEL_FLAG_READY` | Payload is published. Readers must ignore the slot until this is set. |
| 3 | `SURFEL_FLAG_DEAD` | Slot is a hole. The hash must not return it. The payload stays until commit reuses the slot. |

`lock_surfel` tries once, with an atomic compare-exchange. It does not spin. A failed lock means "someone else owns this surfel this frame"; the caller stops. `unlock_surfel` clears the bit with a release, so the payload writes from the critical section are visible first.

## Stats block

Binding 0, eight 32-bit words. The CPU zeros them at init and sets `total_surfels` to `MAX_SURFELS` (65536) and `index_topology_dirty` to 1 so the first frame builds a hash.

```text
word 0  total_surfels           capacity of the whole Surfel array (65536)
word 1  unordered_surfels       how many staging slots were reserved this frame
word 2  live_count              how many committed surfels are not dead
word 3  high_water              exclusive end of the committed region
word 4  global_reserve_counter  used only if POPULATE_VISIBLE_SURFELS_LIST is 1
word 5  discovered_surfels      same
word 6  remaining_holes         dead slots left after commit
word 7  index_topology_dirty    1 if the hash must be rebuilt
```

`live_count` is not "the length of a packed array of surfels". The committed region can contain dead holes. `live_count` is the count of slots in `[0, high_water)` whose `DEAD` bit is clear.

## Two halves

`total_surfels` is 65536. The array is split in half. This is not a spatial split. It is a lifetime split.

```text
index 0                         32768                        65535
|---- committed region ---------|-------- staging region ---------|
     [0, high_water)  live or dead     filled this frame only
     in the hash                      NOT in the hash
```

- The **committed half** is what the hash indexes. Slots here keep their index for the life of the surfel. Killing a surfel sets `DEAD`. It does not slide the tail down.
- The **staging half** is a scratch pad for surfels created during spawn and surfel RT. Slot `32768 + i` is the i-th reservation this frame (`unordered_surfels`). At most `MAX_SURFELS_PER_FRAME` (256) reservations succeed. Next frame, commit copies those records into holes or onto the end of `high_water`, then the staging count is zeroed.

Why the copy exists: a slot that is still named by the current hash must not be overwritten. New surfels are born outside the hash. Commit runs at the start of the next frame, before the hash is rebuilt, and only then do those ids become searchable as committed surfels.

Same-frame lookups of newborns do not use the hash. They scan the dense staging prefix `[32768, 32768 + unordered_surfels)`. That scan is at most 256 records.

## Other buffers

| Binding | Buffer | Contents |
|---|---|---|
| 1 | `surfels` | `Surfel[65536]` |
| 2 | `tree` | `BVHNode` array. **Not written by the live path.** Kept so old shaders still link. |
| 3 | `discovered` | Optional list of visible surfel ids. Unused while `POPULATE_VISIBLE_SURFELS_LIST` is 0. |
| 4 | `surfelOverlappingImage` | `r32ui`, one id per pixel. |
| 5 | `outputImage[2]` | `[0]` indirect, `[1]` direct. `rgba32f`. |
| 7 | `build_words` | Scratch. Layout in chapter 4 and below. |
| 8 | `node_world` | 4096 column-major matrices, one per draw instance. Identity until a scene node moves. |
| 9 | `bone_palette` | 256 skinning matrices. Rigid surfels never read it. |

Binding 6 is not part of this set.

## Scratch words

`build_layout.glsl` is the map. `BUILD_HALF` is 32768. `BUILD_BLOCKS` is 128 (32768 / 256).

```text
word
0                  header (see below)
8                  per-block sums for the prefix pass
8+128              per-block exclusive prefixes
OFF_HOLES          dead slot ids, packed
OFF_IDS_A          live slot ids, packed          <-- hash insert reads this
OFF_KEYS_A         HASH HEADS, then NEXT, then IDS
...                leftover radix / AABB words from the old tree builder
```

Header words that the live path still uses:

| Word | Name | Meaning |
|---|---|---|
| 1 | `HDR_HOLE_COUNT` | Number of ids in `OFF_HOLES`. |
| 3 | `HDR_KEY_COUNT` | `live_count` from the last hash build. The plan compares against this. |
| 7 | `HDR_BUILD_ACTIVE` | 1 if this frame must rebuild the hash. |

`HDR_ROOT`, `HDR_HEAP_P`, and `HDR_HEAP_LEAF0` belong to the unused tree builder.

The hash occupies the old key region:

```text
OFF_HASH_HEAD = OFF_KEYS_A
  3 levels * 8192 buckets of uint heads

OFF_HASH_NEXT = heads + 3*8192
  one uint per live-list slot (not per surfel id)

OFF_HASH_IDS  = next + 32768
  the surfel id stored at that live-list slot
```

`HASH_EMPTY` is `0xFFFFFFFF`. A head with that value has an empty chain. Slot 0 is a valid live-list index, so empty cannot be 0.

## G-buffer

The mesh fragment shader writes, per pixel:

| Target | Contents |
|---|---|
| 0 | World position |
| 1 | World normal |
| 2 | Diffuse albedo |
| 3 | Specular |
| 4 | `uvec4(mesh_id, instance_id, ...)` |
| depth | Hardware depth |

`mesh_id == 0xFFFFFFFF` means sky or a cleared pixel. Discovery and spawn both refuse those pixels. A hash probe from a far-plane sky point would not be wrong so much as useless, and it would still cost the probe.
