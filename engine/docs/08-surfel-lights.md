# 8. Surfel lights

Every surfel is a uniform Lambertian disk light. The displayed value of a surface is `π L_o`, so a pixel shows \(\rho\,(E_\text{dir} + E_\text{ind})\).

## Radiance and power

The publish pass (`gi/publish.comp`) computes, for every alive surfel with at least one direct sample:

\[
L_j = \frac{\rho_j}{\pi}\left(\bar E_{\text{dir},j} + \bar E_{\text{ind},j}\right), \qquad
\text{surfel\_light}_j = (L_j A_j,\ A_j), \qquad
\Phi_j = \text{lum}(L_j A_j).
\]

\(\Phi_j\) (called *flux* in the code) is the luminance of the emitted power divided by π. It is the source weight of the light.

## The geometry term

A uniform Lambertian disk with center `c`, normal \(n_c\), and radius `r`, emitting total power \(L A\) spread over its area \(\pi r^2\), produces irradiance

\[
E_j(x) = L_j A_j \cdot G(c, n_c, r;\ x, n_x)
\]

at a receiver `x` with normal \(n_x\), where `G` is the disk's irradiance per unit radiance, divided by \(\pi r^2\) (`disk_geometry` in `surfels.glsl` and `gi_math.rs`). `G` is 0 behind the disk.

- **Disk entirely above the receiver's horizon**: the closed form of the disk's vector irradiance. With \(h\) the height of `x` over the disk plane, \(\vec\rho\) its in-plane offset, \(T = |x-c|^2 + r^2\), and \(S = \sqrt{T^2 - 4\rho^2 r^2}\):
  \[
  G = -\frac{(n_x\cdot n_c)(S + T - 2\rho^2) + 2h\,(n_x\cdot\vec\rho)}{S\,(S+T)}.
  \]
  It is exact in the near field. The point-light form \(\cos\cos/(d^2+r^2)\) it replaced was off by up to 4× near the disk.
- **Disk cut by the horizon** (a lit floor seen from the wall above it): the 16-gon with the disk's area (circumradius \(1.0130073\,r\)) is clipped to the horizon plane and integrated exactly with Lambert's polygon formula, \(\left|\sum_i \arccos(u_i\cdot u_{i+1})\,(n_x\cdot \hat g_i)\right| / 2\), divided by \(\pi r^2\).

`disk_irradiance_matches_numeric_integration` checks both branches against brute-force integration to 0.2%, including near-field and horizon-cut cases.

Inside the support, `G` is never below `DISK_SUPPORT_FLOOR` (`1e-30`). The support is where part of the disk is in front of it and above the receiver's horizon, and `disk_supports` tests exactly that condition. The clipped polygon can miss a sliver of the true disk; the floor makes "G > 0" and "supported" the same predicate. ReSTIR relies on that (chapter 9). `disk_support_matches_geometry` checks it on 200 000 random configurations.

## Integer flux CDF

Publish builds an inclusive prefix sum over the alive list (this frame's survivors and newborns) of the quantized flux

\[
q_j = \text{clamp}\!\left(\left\lceil \frac{\Phi_j}{\Phi_\text{max,prev}} \cdot 65535 \right\rceil,\ 1,\ 65535\right) \text{ for } \Phi_j > 0, \qquad q_j = 0 \text{ otherwise},
\]

where \(\Phi_\text{max,prev}\) is the previous frame's largest flux (`ST_PREV_MAX_FLUX`). The scan is integer arithmetic: a 1024-wide block scan, a scan of the block totals, and an offset add. The CDF is therefore exactly monotonic, and its total \(Q\) fits in 32 bits because `MAX_SURFELS × 65535 < 2^32`.

Quantization only changes how closely the sampling follows the flux. It never changes correctness, because the stored pmf is the exact selection probability of the integer draw. Every light with \(\Phi > 0\) gets \(q \ge 1\), so no light has zero probability.

## Light tiles

`gi/presample.comp` draws `LIGHT_TILES × LIGHT_TILE_SIZE` = 32 768 lights per frame:

1. Draw an exactly uniform integer `t` in `[0, Q)`: take random bits masked to the next power of two, and reject values ≥ Q (64 tries; the chance of exhausting them is below \(2^{-64}\)).
2. Binary-search the smallest `i` with `cdf[i] > t`. That is \(\lceil\log_2 n\rceil\) reads, 16 for a full pool.
3. Store the light's sphere, id, packed normal, and flux, plus \(\text{pmf} = q_i / Q\).

`integer_cdf_selection_is_exact` checks that every `t` maps to light `i` exactly \(q_i\) times. `masked_rejection_draw_is_uniform` checks the draw. The self-test checks every tile entry against the GPU's own CDF and records.

The ReSTIR pass reads all 8 candidates of a pixel from one tile shared by its 8 × 8 work group, so the 32-byte entries stay in cache.
