# 4. Spatial hash

This is the index. Discovery, spawn, surfel RT, and the VPL pass all find committed surfels through it.

A bounding-volume hierarchy was the previous index. It is not built anymore. Chapter 10 lists the files that still describe it.

## Why a hash

A surfel is a sphere whose radius grows with distance from the camera, from 10 to 120 world units. A tree of those spheres has fat boxes. A point query in an open courtyard visits a long chain of nodes and almost never rejects a child. That walk is random, and neighboring pixels do not share it.

The hash stores each center in one cell. A query loads the cells around the point. Neighboring pixels share those cells, so the loads hit cache. Insert is one atomic per live surfel, and only on frames where the set changed.

## Levels

Large and small spheres do not share a cell size. A cell must be at least as big as the radius, or a sphere that contains the query point can sit more than one cell away from that point. Three levels cover the legal radius range:

| Level | Radius stored here | Cell size |
|---|---|---|
| 0 | `radius < 32` | 32 |
| 1 | `32 ≤ radius < 64` | 64 |
| 2 | `radius ≥ 64` (max 120) | 128 |

The cell of a center is `floor(center / cell_size)`, component-wise. Because the cell is at least the radius, any point inside the sphere is in the same cell as the center or in one of the 26 adjacent cells. The query therefore probes a 3×3×3 cube and no further.

```text
cell size 32, radius 20

          +----+----+----+
          |    |    |    |
          +----+----+----+
          |    | C  |  P |
          +----+----+----+
          |    |    |    |
          +----+----+----+

C = cell of the center
P = a point inside the sphere, one cell away
The 3x3x3 around P includes C.
```

A point outside the sphere can still land in that neighborhood. The probe always tests `point_inside_surfel` (squared distance against the stored radius) before accepting an id.

## Buckets

There are 8192 buckets per level. The bucket of a cell is

```text
hash = (x * 73856093) xor (y * 19349663) xor (z * 83492791)
bucket = hash & 8191
```

8192 is a power of two, so the mask is exact. Different cells can share a bucket. That is a chain, not a drop. Nothing is evicted when a bucket is busy.

## Chains

Each live surfel occupies one node. The node index is the surfel's index in `OFF_IDS_A` (0 .. live_count-1), not the surfel slot.

```text
level 1, bucket 42

head[1][42] = 7
                 |
                 v
            next[7] = 2          ids[7] = surfel slot 100
                 |
                 v
            next[2] = 0xFFFFFFFF ids[2] = surfel slot 14
```

`head` is `0xFFFFFFFF` when the bucket is empty. `next[slot]` is the previous head, so the newest insert is at the front. After the insert dispatch finishes, a compute barrier runs before any query. Queries never race the insert.

Walks stop after 32 hops. With a few thousand surfels and 8192 buckets the chains are short. A hop limit that is too low silently misses surfels. Raising it costs more only on the buckets that are actually long.

## Build

`surfel_keys.comp`. The file name is from the Morton-key pass it replaced. The push constant `phase` selects the step. Both steps return immediately if `HDR_BUILD_ACTIVE` is 0.

**Phase 0, clear.** 96 groups of 256 threads (`3 * 8192 / 256`). Each thread writes `HASH_EMPTY` into one head. Next and id arrays are not cleared. Insert overwrites the nodes it uses, and queries only follow heads.

**Phase 1, insert.** 128 groups of 256, which covers `BUILD_HALF`. Thread `gid` does the work only when `gid < live_count`.

```text
surfel_id = OFF_IDS_A[gid]
level     = surfel_level(surfels[surfel_id].radius)
cell      = floor(center / cell_size(level))
prev      = atomicExchange(head[level][bucket(cell)], gid)
next[gid] = prev
ids[gid]  = surfel_id
```

Thread 0 also stores `HDR_KEY_COUNT = live_count` and clears `index_topology_dirty`. The next quiet frame will skip the rebuild.

`atomicExchange` needs a coherent buffer. The scratch binding in this shader is declared `coherent`. Do not drop that qualifier.

## What the hash does not store

- Staging surfels. They are scanned as a dense list of at most 256.
- Dead surfels. They are absent from `OFF_IDS_A`. A query still checks `DEAD`, in case a stale chain is read.
- Bounds. There is no AABB. Rejection is the sphere test on the payload.
- Radius edits that happen in mark without setting `index_topology_dirty`. The surfel stays in the level it had at insert time. Chapter 11.

## Memory traffic, compared with the tree

| | Tree walk that used to run | Hash |
|---|---|---|
| Build | sort every live key, build topology, refit bounds | clear 24k heads, one atomic per live surfel |
| Query | up to 64 random 64-byte nodes, divergent per pixel | up to 3 × 27 chain heads, shared by neighbors |
| Quiet frame | skipped, if the plan said so | skipped, same plan |
