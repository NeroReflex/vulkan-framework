# 11. Testing

Four layers of checks. Unit tests cover the math on the CPU. A self-test checks the GPU's data structures from a readback. Captures compare the image against a path-traced reference. Timestamps measure the cost. Run GPU tests with `env -u LD_PRELOAD` on the development laptop, where `libnvidia-hide.so` hides the RTX 4080.

## Shaders

```bash
engine/shaders/gi/validate.sh
```

This compiles every GI shader the way `inline_spirv!` does: it adds the `#version 460` header and targets Vulkan 1.2. Then it runs `spirv-val`. It catches invalid SPIR-V that the compiler accepts (chapter 12) before the driver does.

## Unit tests

```bash
cargo test --lib gi_math
```

`engine/src/rendering/pipeline/gi_math.rs` mirrors the shader math and checks:

- grid: each level's cell holds the surfel diameter, a surfel spans at most 8 cells, and a point query finds every surfel containing the point;
- area: the Monte Carlo partition-of-unity estimate integrates to the disk area for a lone surfel, and overlapping surfels sum to the area of their union;
- the disk geometry term against brute-force integration (13 cases, including horizon cuts), and its support test (200 000 random cases);
- the integer CDF draw: exact selection probabilities and a uniform rejection draw;
- RIS, and Z-normalized merging of reservoirs from domains with different supports, against the exact sum on a toy light set;
- free-stack pops with interleaved failures hand out unique slots.

## Self-test

`ART_RTIC_SELFTEST=f1,f2,...` copies the whole GI state back to the host at the end of the listed frames and checks it (`gi_selftest.rs`). The process exits with status 2 if any check fails.

- **Pool**: alive + free = `MAX_SURFELS`; every slot is alive or free exactly once; alive surfels carry `ALIVE`, free ones do not; radii are in range.
- **Grid**: the CSR table equals a CPU rebuild over the survivors; the level mask is right; no stray items.
- **Lights**: every record matches its surfel (generation, flux = `lum(L·A)`, sphere, normal); free slots have flux 0; area ≤ πr²; `ST_MAX_FLUX` is the true maximum.
- **CDF and tiles**: each \(q_i\) is in range and within one unit of the CPU quantization; the total is the CDF end; every tile entry's pmf equals its CDF step over `Q` and its data matches the record.
- **Images**: overlay ids are alive; no negative or non-finite irradiance; coverage ≥ 85% of surface pixels after frame 120.

## Captures and the reference

`ART_RTIC_CAPTURE=dir@f1,f2` (default frame 600) turns on the reference path tracer and writes, for each listed frame:

- `fNNNNN_e_dir.pfm`, `_e_ind.pfm`, `_e_ref.pfm`, `_albedo.pfm`: 32-bit float images;
- `_e_ind_frames.pfm`: how many frames each pixel's accumulated indirect light averages;
- `_final.jpg`, `_reference.jpg`, `_indirect.jpg`, `_reference_indirect.jpg`, `_surfels.jpg`: Reinhard-tonemapped previews.

The reference (`gi/reference.rgen`) is a Lambertian path tracer with cosine sampling and next-event estimation of the directional lights:

\[
E_\text{ind}(x) = \pi L(h_1), \qquad L(h) = \frac{\rho}{\pi}\big(E_\text{dir}(h) + E_\text{ind}(h)\big) \;\Rightarrow\; E_\text{ind}(x) \approx \sum_k \Big(\prod_{j<k}\rho_j\Big)\rho_k\,E_\text{dir}(h_k).
\]

It takes 2 samples per pixel per frame, with 8 bounces under multibounce and 1 without, and averages progressively while the view is still. It uses the same hit shader, albedo, and lights as the surfel pipeline, and it shares no code with the cache.

Pin the camera with `ART_RTIC_CAMERA="x,y,z,horizontal,vertical"`. Input and wander are then ignored. The default pose is `152,650,-8.5,-17.25,-0.03`.

```bash
env -u LD_PRELOAD ART_RTIC_NO_UI=1 ART_RTIC_MAX_FRAMES=620 ART_RTIC_GI_DISABLE=multibounce \
    ART_RTIC_CAPTURE=/tmp/gi/cap_1b@600 ART_RTIC_SELFTEST=600 cargo run --bin artrtic
cargo run --bin artrtic-compare -- /tmp/gi/cap_1b f00600 --max-bias 3 --max-block-rmse 30
```

`artrtic-compare DIR TAG` reports, on surface pixels:

- the bias of the mean indirect luminance;
- the relative RMSE per pixel and on 16 × 16 block means;
- percentiles of the block ratio estimate/reference;
- a heatmap `TAG_ratio.jpg` of the log ratio (blue too dark, red too bright).

`--max-bias` and `--max-block-rmse` turn the numbers into gates, with exit status 1 on failure. The gates in use are 3% bias and 30% block RMSE for one bounce, and 5% and 25% for multibounce. Pixel RMSE mostly measures the reference's own noise and is not gated.

## Timings

`ART_RTIC_TIMINGS=1` prints per-pass GPU times averaged over 120 frames:

```text
GI GPU ms (avg of 120): total ... | maintain | grid | surfel update | coverage+spawn | light cdf | restir temporal | restir spatial | shade | accumulate | reference | copies
```

`ART_RTIC_BENCH_WANDER=1` moves the camera along a fixed path for drift and performance runs. `ART_RTIC_MAX_FRAMES=N` exits after N frames and prints frame-time percentiles. With vsync the GPU clocks down between frames, which inflates pass times. Add `ART_RTIC_STOP_AFTER=hdr` to skip present and run unthrottled.

## Measured results

RTX 4080 Laptop GPU, Sponza, 1920 × 1080 render area. Accuracy runs use the debug build with the validation layer on, except the 4000-frame run; timing runs use the release build. All runs keep the default "converge when still".

| Run | Bias | Block RMSE | Blocks within 10% | Median block ratio |
|---|---|---|---|---|
| default pose, 1 bounce, frame 600 / 1200 | +0.7% / +0.1% | 23% | 89–91% | 1.03–1.04 |
| default pose, multibounce, frame 600 / 1000 / 4000 (release) | -0.3% / -0.6% / -1.1% | 15% | 90–91% | 1.01–1.02 |
| pose `-500,250,0,-14.1,0.25`, 1 bounce, frame 600 / 1200 | -0.1% / -0.6% | 8.2% | 96% | 1.01 |
| same pose, multibounce, frame 600 / 1200 | -1.3% / -1.7% | 8.0% | 91% | 1.00 |

Convergence, default pose with multibounce: the cache stops changing about 250 frames after the camera stops. From then on, the change of the image between two captures shrinks as \(1/\sqrt{N}\): 9.2% per pixel (1.0% on 16 × 16 blocks) from frame 1000 to 2000, and 6.1% (0.5%) from 2000 to 4000. The reference's own changes over the same intervals are 3.6% and 2.5%. The block error against the reference stays at 15% because it is the cache's error, not noise.

That error is concentrated in the near field of thin, intensely lit features, such as the wall just above the lit top of the default view's arches. The surfel's disk emits the right power but spreads it over its whole radius, so the glow is softer and dimmer than in the reference. Fixing it would take a per-surfel emission centroid, which means a larger surfel record.

Self-test passes at frames 5, 150, 300, 600, and 1200 at fixed poses, and at 300, 900, 1500, and 1790 during a wander run. Over that run the pool stays between 46 000 and 57 000 surfels with 0 failed spawns, and coverage stays ≥ 99.9% once warmed up. The validation layer reports no errors.

Over a 1500-frame wander, unthrottled, the GI section costs 6.1–8.7 ms. The whole frame averages 9.5 ms (105 FPS), with p99 at 11.3 ms and no drift. The biggest passes are spatial reuse (1.2–1.8 ms), coverage (1.0–1.4 ms), surfel update (1.3–1.4 ms), temporal reuse (1.0–1.8 ms), and shade (0.9 ms). The grid build is 0.07 ms, and light CDF plus presampling is 0.06 ms. With vsync the frame is locked at 60 FPS, with the maximum frame at 18.5 ms.
