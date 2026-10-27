# 5. Queries

All surfel lookups weigh surfels with the same kernel. The kernel decides which surfels cover a point and how much each one counts.

## The kernel

For a surfel with center `c`, normal `n_c`, and radius `r`, evaluated at a point `p` with normal `n_p` (`surfel_kernel` in `gi/common.glsl`):

\[
k(p) = \left(1 - \frac{|p-c|^2}{r^2}\right)^2 \quad\text{when } |p-c| < r,\ n_p \cdot n_c \ge 0.5,\ |(p-c)\cdot n_c| \le 0.25\,r,
\]

and 0 otherwise. The normal test keeps surfels from leaking around corners. The plane test keeps a surfel on a floor from claiming the underside of a thin slab above it.

Weights are normalized wherever they are used, so overlapping surfels form a **partition of unity**: the fraction of a point that belongs to surfel `j` is \(k_j(p) / \sum_k k_k(p)\).

## Coverage of a pixel

Coverage (chapter 6) sums the kernels at each pixel's primary hit. The sum is what the coverage view shows: red below 0.5, green near 1, blue above 1.5. The surfel with the largest kernel is the pixel's **dominant** surfel. It goes into `surfel_overlay` and gets its `last_seen` refreshed.

## Area of a surfel

A surfel's area estimate is its partition-of-unity share of the surface (chapter 7). Take a uniform point `q` on the disk. If `q` lies on the surfel's own surface, then

\[
a = \pi r^2 \cdot \frac{k_\text{self}(q)}{\sum_k k_k(q)},
\]

otherwise `a = 0`. The expectation of `a` is \(\int_{\text{surface} \cap \text{disk}} k_\text{self} / \sum k\). Summed over all surfels, that is the covered surface area, with no double counting where surfels overlap. The self-test checks that `area ≤ πr²` for every surfel.

## Radiance of the cache

`cache_radiance(p, n)` returns the kernel-weighted mean radiance of the surfels covering `p`:

\[
L_\text{cache}(p) = \frac{\sum_j k_j(p)\, L_j}{\sum_j k_j(p)},
\]

counting only surfels that are already lights (`A > 0`). It also returns the total weight `w_sum`, the weight of lit surfels `w_lit`, and the dominant surfel. A surfel's bounce ray uses it to read the outgoing light where it lands (chapter 7). `w_lit = 0` means the cache has nothing there.

## Cost

Each query reads at most one bucket per occupied level (chapter 4) and evaluates the kernel of each candidate, using `geo[]` and `nrm[]` (20 bytes per candidate). The full 128-byte record is never read in a query loop. Coverage runs one query per pixel. Each surfel update runs one query for its area sample, one for its bounce hit, and up to three more for path continuation.
