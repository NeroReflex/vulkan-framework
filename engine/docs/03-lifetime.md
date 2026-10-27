# 3. Lifetime

A surfel is a slot index into `surfels[]`. It is allocated from a free stack, listed in an alive list every frame it survives, and pushed back on the stack when it is evicted. Its payload never moves, so nothing that stores an id (overlay, reservoirs, light tiles) needs fixing up.

## Free stack

`free_ids[0 .. state[ST_FREE])` is a stack of free slots.

- **Pop** is `surfel_alloc()`: `atomicAdd(state[ST_FREE], -1)`, then read `free_ids[prev - 1]`. When the counter was already 0, the pop puts the 1 back, counts `ST_SPAWN_FAILED`, and returns `INVALID_ID`. Only the surfel update and coverage passes pop.
- **Push** happens only in maintain: `free_ids[atomicAdd(state[ST_FREE], 1)] = id`.

No pass both pushes and pops, so within a pass the counter only goes down, apart from undoing failed pops. Once it has reached 0 it never becomes positive again, because each undo only cancels that pop's own decrement. Successful pops therefore see distinct positive counter values, and so distinct slots.

`pool_reset.comp` fills the stack with every slot and zeroes every state word. It runs on the first frame and whenever `gi_reuse_frames` is 0, which is also when the image history is cleared.

## Alive lists

`alive_ids` holds two lists of `MAX_SURFELS` ids. `alive_in_base()` and `alive_out_base()` pick them by `frame_index` parity.

1. **Frame begin** copies `ST_ALIVE_OUT` into `ST_ALIVE_IN` and zeroes the per-frame counters.
2. **Maintain** walks the input list. Survivors are appended to the output list with `surfel_push_alive`.
3. **Grid build** phase 0 freezes `ST_SURVIVORS = ST_ALIVE_OUT`. The grid and the surfel update launch cover exactly those ids.
4. **Surfel update** and **coverage** append newborns to the output list behind the survivors. Newborns are part of this frame's light CDF but not of this frame's grid. They become searchable next frame.

## Birth

`surfel_init()` writes a complete record and its mirrors. It bumps `generation`, sets `area = πr²` with `area_n = 0` (the first area sample replaces it), clears `surfel_light` and `light_rec`, and pushes the id on the output list. A screen-spawned surfel starts with `e_dir_n = 0`: it is not a light until its first update has measured its direct light. A ray-spawned surfel is born with the direct light measured at the hit, so it is a light in the same frame.

## Eviction

Maintain keeps a surfel only if all of these hold:

- `ALIVE` is set and `DOOMED` is not (coverage sets `DOOMED`, chapter 6);
- its instance still exists with the same `instance_generation` (chapter 13);
- its new world center is within `SURFEL_MAX_DISTANCE` (6000) of the eye;
- `frame_index - last_seen` is at most `SURFEL_UNSEEN_FRAMES` (240). Under pool pressure the limit is `SURFEL_PRESSURE_UNSEEN_FRAMES` (30). Pressure means fewer than `MAX_SURFELS / 8` free slots at frame begin. A settled cache (chapter 9) skips this rule.

An evicted surfel has its flags cleared, its light state emptied, and its slot pushed on the free stack.

## Being seen

`last_seen` is the only thing that keeps a surfel alive, and two passes write it.

- **Coverage** sets `last_seen = frame` on the dominant surfel of every fourth pixel in each direction. A surfel touched within `SURFEL_RECENT_FRAMES` (2) counts as *on screen*.
- **Surfel update**: when an on-screen surfel's bounce ray hits a surface, the dominant surfel at the hit gets `last_seen = frame - 3` unless it is already more recent. That keeps the lights of visible surfaces alive without marking them as on screen. If they counted as on screen, their own rays would spawn more offscreen surfels and the cache would grow without bound.

## Generations

`generation` changes on every allocation and is never reset, not even by a pool reset. Reservoirs and light records carry it. `reservoir_light()` returns an empty record unless the stored generation equals the light record's, and evicted slots have their records cleared. So a reservoir that points at a recycled slot is simply invalid, never wrong.
