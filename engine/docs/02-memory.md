# 2. Memory

Every GI pass binds the same surfel descriptor set. It is declared once in `engine/shaders/gi/surfels.glsl` and built in `GILighting::new`. Compute passes see it as set 2 (`gi/compute.glsl`) and ray generation passes as set 3. The status block is set 0 or 1.

## The status block

`gi_data` (`status.glsl`, std140) is written by `GiDriver::begin_frame` (`gi_driver.rs`) with `update_buffer` at the start of every frame.

| Field | Meaning |
|---|---|
| `gi_reuse_frames` | frames since the history was invalidated (chapter 1) |
| `frame_index` | frame counter; parity selects ping-pong buffers |
| `still_frames` | frames with an unchanged pose and no skinned clip playing; 0 restarts the reference |
| `flags` | `GI_FLAG_*`: multibounce, reference, temporal, spatial, capture, reset, spawn, accumulate, converge |
| `reference_bounces` | 8 with multibounce, 1 without |
| `render_width`, `render_height` | render area (1920 × 1080) |
| `pixel_angle` | angle one pixel subtends at the image center: `2 atan(1 / P[1][1]) / H` |
| `eye_position` | camera position |
| `prev_view`, `prev_proj` | last frame's matrices, used for reprojection |

## The surfel record

`struct Surfel` is 128 bytes (`SURFEL_STRIDE`), and there are `MAX_SURFELS` = 65536 slots. A record never moves. Passes that only need the shape read two compact mirrors instead: `geo[id] = (center, radius)` and `nrm[id]`, the octahedral world normal. Maintain refreshes both every frame.

| Field | Written by | Meaning |
|---|---|---|
| `position`, `normal` | maintain, init | world center and normal this frame |
| `radius` | init | fixed at birth |
| `bind_position`, `normal_bind` | init | center and normal in the instance's space |
| `instance_id`, `instance_generation`, `bone` | init | what the surfel is attached to (chapter 13) |
| `generation` | init | bumped every time the slot is allocated |
| `albedo` | init, update | mean albedo of the surface under the disk (EMA, shares `area_n`) |
| `area`, `area_n` | update | partition-of-unity area (EMA) and sample count |
| `e_dir`, `e_dir_n` | update | direct irradiance (EMA) and sample count |
| `e_ind`, `e_ind_n` | update | indirect irradiance (EMA) and sample count |
| `flags` | init, coverage, maintain | `ALIVE`, `DOOMED` |
| `last_seen` | coverage, update | last frame it mattered (chapter 3) |
| `spawn_frame` | init | birth frame |
| `lit_frame` | publish | first frame it was a light (`INVALID_ID` before) |

## State words

`state[]` is an array of `int` counters. Its first 17 words are named `ST_*` in `surfels.glsl` and mirrored in `gi_math.rs`.

| Word | Meaning |
|---|---|
| `ST_ALIVE_IN`, `ST_ALIVE_OUT` | length of last frame's alive list and of this frame's |
| `ST_SURVIVORS` | how many of this frame's ids came from maintain; the grid and the update pass cover exactly these |
| `ST_FREE`, `ST_FREE_AT_BEGIN` | free stack height, now and at frame begin |
| `ST_SPAWN_SCREEN`, `ST_SPAWN_RAY`, `ST_SPAWN_FAILED` | spawn attempts this frame, and pops that found the stack empty |
| `ST_EVICTED`, `ST_DOOMED` | evictions and prune marks this frame |
| `ST_LEVEL_MASK` | bit `l` set when grid level `l` holds a surfel |
| `ST_GRID_ITEMS` | total bucket memberships |
| `ST_TOTAL_FLUX`, `ST_LIGHT_COUNT` | end of the flux CDF, and how many entries it has |
| `ST_MAX_FLUX`, `ST_PREV_MAX_FLUX` | largest light flux this frame and last frame (float bits) |
| `ST_STABLE_FRAMES` | frames in a row without allocations, evictions, or dooms (chapter 9) |

## Light state

Publish turns each surfel into a light (chapter 8). It writes three things.

- **`surfel_light[id] = (L·A, A)`**: emitted radiance times area, and the area. It is kept even when the radiance is 0, because a dark surfel is still a valid cache entry.
- **`light_rec[id]`**, a 32-byte `LightRecord`. It holds `sphere = geo[id]`, plus `info = (packed normal, flux bits, generation, lit_frame)` with flux `= luminance(L·A)`. This is all that ReSTIR reads about a light. A record with flux 0 is not a light. Records are cleared by `empty_light_record()` on init, eviction, and pool reset, so a stale reservoir never finds a matching generation.
- **`flux_cdf[i]`**: inclusive prefix sum of quantized flux over the alive list.

Presample fills **`light_tiles`**: 128 tiles × 256 `TileEntry` of 32 bytes. Each entry holds the light's sphere, id, packed normal, flux, and the exact probability with which it was drawn.

## Grid

`grid_count`, `grid_start` (`GRID_BUCKETS + 1` entries), `grid_cursor`, and `grid_items` (at most `8 × MAX_SURFELS` ids) form a CSR table over 2^18 hashed buckets (chapter 4). `scan_blocks` is scratch for the grid and flux scans.

## Reservoirs

`reservoirs[]` holds three regions of `W × H` entries, each 32 bytes (`RESERVOIR_STRIDE`): `{id, gen, W, M, pos, packed normal}`. Two regions ping-pong between frames and hold the final, spatially reused reservoirs; `res_current()` selects one by `frame_index` parity. The third region, `RES_TEMP`, holds the temporal result of the current frame before spatial reuse. `M = 0` marks a pixel with no surface.

## Bindings

| Binding | Contents |
|---|---|
| 0 | `state[]` |
| 1 | `surfels[]` |
| 2 | `free_ids[]` |
| 3 | `alive_ids[]`, two lists of `MAX_SURFELS` |
| 4, 5 | `geo[]`, `nrm[]` |
| 6 | `surfel_light[]` |
| 7–10 | grid count, start, cursor, items |
| 11, 12 | `flux_cdf[]`, `scan_blocks[]` |
| 13 | `light_tiles[]` |
| 14 | `reservoirs[]` |
| 15–18 | node matrices, bone palette, instance generations, instance bone bases (read only) |
| 19 | `surfel_overlay` (r32ui) |
| 20–26 | `gi_images[7]` (rgba32f) |
| 27 | `light_rec[]` |

The light records sit at 27, not 21. The framework's descriptor writer gives each element of an image array its own consecutive binding slot, so `gi_images[7]` really occupies 20 through 26.
