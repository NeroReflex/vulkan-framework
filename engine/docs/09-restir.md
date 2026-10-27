# 9. ReSTIR

Each pixel estimates its indirect irradiance

\[
E_\text{ind}(x) = \sum_j E_j(x)\,V(x, y_j)
\]

by selecting one surfel light per frame with resampled importance sampling and reusing selections across frames and neighbours. The background is in the [gentler introduction to ReSTIR](https://interplayoflight.wordpress.com/2023/12/17/a-gentler-introduction-to-restir/).

## Definitions

- **Source pmf**: \(p(j) = q_j / Q\), exactly the probability with which a light tile entry holds light `j` (chapter 8).
- **Target**: \(\hat p_x(j) = \Phi_j\,G_j(x) = \text{lum}(E_j(x))\), the unshadowed irradiance luminance (`record_target`).
- **Reservoir** `{y, gen, W, M, pos, n}`: selected light and its slot generation, unbiased contribution weight, candidate count, and the surface it was built for.
- **Estimate**: \(E_\text{ind}(x) \approx E_y(x)\,V(x, y)\,W\).

## Initial candidates and temporal reuse

`gi/restir_temporal.comp` runs on 8 × 8 work groups. A group picks one light tile by hashing its group id with the frame. Each pixel then draws `RESTIR_INITIAL_CANDIDATES` (8) entries from that tile and runs weighted reservoir sampling with weights \(w = \hat p_x(j) / p(j)\).

Temporal reuse then reprojects `x` with `prev_view` and `prev_proj`. It accepts the previous reservoir at the reprojected pixel only if that surface is similar (`similar_surface`): normals within `cos ≥ 0.9`, and plane distance within 2% of the eye distance. The previous reservoir is looked up again through `reservoir_light()`, so if its slot was recycled (generation mismatch) or it is not a light any more, its target is 0. It joins with weight \(\hat p_x(y_\text{prev})\,W_\text{prev}\,M_\text{prev}\), where \(M_\text{prev}\) is capped at 20 × 8.

**Z-normalization.** The final weight is \(W = w_\text{sum} / (Z\,\hat p_x(y))\), where `Z` sums the `M` of every domain that could have produced `y`:

- the current candidates always count (8), because a selected `y` is a light now and has \(\hat p_x(y) > 0\);
- the previous domain counts only if `y` was already a light in the previous frame (`lit_frame < frame`) and its target at the previous surface is positive (`record_supports`).

A plain \(1/M\) instead of \(1/Z\) would be biased wherever a light is visible from one surface and not from the other, as at every concave corner. The comparison in chapter 11 confirms that the reuse is unbiased: plain RIS without reuse and full reuse converge to the same mean.

## Spatial reuse

`gi/restir_spatial.comp` merges the pixel's temporal reservoir with `RESTIR_SPATIAL_NEIGHBORS` (4) reservoirs drawn uniformly from a 30-pixel disk. Each neighbour must pass the same similar-surface test. Each candidate is re-targeted at the center pixel, \(\hat p_x(y_k)\,W_k\,M_k\). `Z` sums the `M` of the domains (center included) whose own surface supports the selected `y`.

`Z` only needs to know *whether* each domain's target is positive, so it calls the cheap `record_supports` instead of evaluating the geometry term five times. That is valid because the target's support is exactly that predicate (chapter 8).

Spatial reuse reads `RES_TEMP` and writes the current ping-pong region. The temporal pass of the next frame reads that region back as "previous".

## Shading

`gi/restir_shade.rgen`, one invocation per pixel:

- \(E_\text{dir}(x)\): directional lights with one shadow ray each, written to `IMG_EDIR`.
- \(E_\text{ind}(x) = E_y(x) \cdot W \cdot V\), where `V` is one ray from `x` to the selected surfel's center, offset along the surfel's normal. It is written to `IMG_EIND_RAW`. Using the center for the visibility of a whole disk is an approximation that holds as long as disks are small compared with occluder features. Coverage keeps them at about 24 pixels.

## Accumulation

`gi/accumulate.comp` keeps a running mean of the shaded indirect light over at most `ACCUM_MAX_HISTORY` (32) frames. The history is reprojected the same way as the reservoirs and is dropped on a surface mismatch. The count is stored in the image's alpha. The result is `IMG_EIND`, which final shading reads and which is copied to `IMG_EIND_HISTORY` at the end of the frame.

## Switches

`GI_FLAG_TEMPORAL`, `GI_FLAG_SPATIAL`, and `GI_FLAG_ACCUMULATE` turn the three reuse steps off one at a time (UI or `ART_RTIC_GI_DISABLE`). With all three off, the estimate is plain 8-candidate RIS every frame: unbiased, but noisy.

## Cost

At 1920 × 1080, temporal takes about 1.0–1.3 ms, spatial 1.2–1.7 ms, shade 0.9 ms, and accumulate 0.6 ms. The geometry term dominates: 8 evaluations for the candidates, 1 for the previous reservoir, and 4 for the neighbours.
