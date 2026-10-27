# AGENT.md

Guide for agents (and humans) working on this repository: how it is laid out, how to build and run it, how changes to the renderer are tested and validated, and the traps that have already cost time once.

The design itself is documented in [`engine/docs/`](engine/docs/README.md). Read the chapter for the area you touch before changing it; chapter 12 (`12-maintenance.md`) lists the invariants the passes rely on.

## Repository

Cargo workspace (`Cargo.toml` at the root):

| Crate | What it is |
|---|---|
| `vulkan_framework` | Rust Vulkan wrapper over `ash` (devices, memory, pipelines, samplers, ray tracing). |
| `engine` (package `artrtic`) | The renderer: G-buffer, surfel GI with ReSTIR, reference path tracer, skinning, ImGui editor. Binaries `artrtic` and `artrtic-compare`. |
| `artrtic_imgui` | Dear ImGui integration. |
| `vulkan_framework_sdl2_glue`, `vulkan_framework_glfw_glue` | Windowing glue. |
| `cooker` | Asset cooking. |
| `examples` | Framework examples. |

Engine layout:

- `engine/src/rendering/system.rs`: frame recording, barriers, swapchain, device creation.
- `engine/src/rendering/pipeline/global_illumination.rs`: GI resources and pass recording (`GILighting`).
- `engine/src/rendering/pipeline/gi_math.rs`: CPU mirrors of the shader math and constants, plus unit tests.
- `engine/src/rendering/gi_driver.rs`: per-frame status block, `still_frames`, capture and self-test scheduling.
- `engine/src/rendering/gi_selftest.rs`, `gi_capture.rs`: GPU state checks and image captures.
- `engine/src/rendering/debug_state.rs`, `engine/src/ui/mod.rs`: GI switches, view modes, debug panel.
- `engine/shaders/config.glsl`: every tunable constant. `engine/shaders/gi/*`: the GI passes.
- `scene.json` + `sponza_from_obj.tar` at the root: the default scene. Run binaries **from the repository root** or the scene is not found.

## Hard rules

- **GPU access.** The development laptop preloads `libnvidia-hide.so` to keep the RTX 4080 powered off. Every GPU run must be prefixed with `env -u LD_PRELOAD`. Never remove or edit `/home/denis/.lib/libnvidia-hide.so` or the global preload.
- **Performance constraints from the owner.** No runtime reordering or sorting of surfels; slots are stable for life. Every lookup must be O(1) or at worst O(log n) (the hashed grid is O(1), the light CDF search O(log n)).
- **Correctness bar.** Do not land something that has not been measured against the reference path tracer, or that is not mathematically sound. Report measured numbers honestly; never loosen a gate to make a change pass.
- **Plan files** under `~/.cursor/plans/` are not to be edited when implementing them.
- Do not commit unless asked. The owner commits.

## Build

```bash
cargo build --release          # timing and long accuracy runs
cargo build                    # debug: Vulkan validation layer on automatically
cargo test --release --lib     # all unit tests (gi_math mirrors included)
engine/shaders/gi/validate.sh  # compile + spirv-val every GI shader
```

**Shaders are embedded with `inline_spirv!`, which does not track `#include`s.** After editing any `.glsl` header (or any shader at all, to be safe), touch the Rust files that embed them, or the old SPIR-V stays in the binary:

```bash
touch engine/src/rendering/pipeline/global_illumination.rs engine/src/rendering/pipeline/mesh_rendering.rs
```

`validate.sh` covers the GI shaders and `final_rendering.frag`, not `mesh_rendering.frag`; mesh shader errors show up at build time.

## Running

```bash
env -u LD_PRELOAD ./target/release/artrtic
```

Environment switches (all read in `engine/src`):

| Variable | Effect |
|---|---|
| `ART_RTIC_CAMERA="x,y,z,horizontal,vertical"` | Pin the camera; input and wander are ignored. Default pose `152,650,-8.5,-17.25,-0.03`; second test pose `-500,250,0,-14.1,0.25`. |
| `ART_RTIC_MAX_FRAMES=N` | Exit after N frames and print frame-time percentiles. |
| `ART_RTIC_NO_UI=1` | Skip ImGui. |
| `ART_RTIC_STOP_AFTER=none\|mesh\|gi\|final\|hdr\|ui\|all` | End the command buffer after a stage; anything but `all` skips acquire/present and runs unthrottled. `hdr` is the standard for benchmarks and long runs. |
| `ART_RTIC_CAPTURE=dir@f1,f2` | Turn on the reference and write PFM/JPG captures at those frames. |
| `ART_RTIC_SELFTEST=f1,f2` | Read back the whole GI state at those frames and check invariants; exit status 2 on failure. |
| `ART_RTIC_TIMINGS=1` | Per-pass GPU times, averaged over 120 frames. |
| `ART_RTIC_BENCH_WANDER=1` | Fixed camera path for drift and performance runs. |
| `ART_RTIC_GI_DISABLE=multibounce,temporal,spatial,accumulate,spawn,converge` | Turn GI switches off. |
| `ART_RTIC_GI_REFERENCE=1` | Turn the path-traced reference on. |
| `ART_RTIC_VIEW=<index>` | Start in a view mode (table in `engine/docs/10-composite.md`: 0 final … 5 reference, 7 difference). |
| `ART_RTIC_NO_VALIDATION=1` | Debug build without the validation layer. |
| `ART_RTIC_NO_PRESENT=1` | Diagnostic only; runs at ~1 FPS. Use `STOP_AFTER=hdr` for unthrottled runs instead. |
| `ART_RTIC_FRAMES_IN_FLIGHT`, `ART_RTIC_QUEUE_COUNT`, `ART_RTIC_PREVIEW` | Experiments: frames in flight, queue count, preview. |

## How changes are tested

Use the layers in order; each catches a different class of bug. Commands assume the repository root.

### 1. Unit tests (CPU)

```bash
cargo test --release --lib gi_math
```

`gi_math.rs` mirrors the shader math (grid, partition-of-unity area, disk geometry term vs brute force, support test, integer CDF, RIS and Z-normalized merging, free stack). When you change shader math, change the mirror and keep the tests passing. Constants marked `rust:` in `config.glsl`/`surfels.glsl` must match `gi_math.rs`.

### 2. Shader validation

```bash
engine/shaders/gi/validate.sh
```

Catches SPIR-V that glslang accepts but the driver rejects.

### 3. Self-test (GPU state invariants)

```bash
env -u LD_PRELOAD ART_RTIC_NO_UI=1 ART_RTIC_CAMERA="152,650,-8.5,-17.25,-0.03" \
  ART_RTIC_MAX_FRAMES=1220 ART_RTIC_SELFTEST=150,300,600,1200 ./target/debug/artrtic
```

Checks pool accounting, the grid against a CPU rebuild, light records, the flux CDF and tiles, and image sanity. Each line also prints alive/spawned/evicted/doomed counts and `unchanged for N frames` (the stable-cache counter), which is the quickest way to see pool dynamics.

### 4. Accuracy against the reference

```bash
env -u LD_PRELOAD ART_RTIC_NO_UI=1 ART_RTIC_CAMERA="152,650,-8.5,-17.25,-0.03" \
  ART_RTIC_MAX_FRAMES=1220 ART_RTIC_CAPTURE=/tmp/gi/cap@600,1200 ART_RTIC_SELFTEST=600,1200 \
  ./target/debug/artrtic
./target/release/artrtic-compare /tmp/gi/cap f00600 --max-bias 5 --max-block-rmse 25
```

Gates in use (exit status 1 on failure):

| Mode | `--max-bias` | `--max-block-rmse` |
|---|---|---|
| one bounce (`ART_RTIC_GI_DISABLE=multibounce`) | 3 | 30 |
| multibounce | 5 | 25 |

Run both modes at both poses. Pixel RMSE is mostly the reference's own noise and is not gated; look at block RMSE, the block-ratio percentiles, and the `_ratio.jpg` heatmap (blue too dark, red too bright). Accuracy runs use the debug build so the validation layer is on; check the log has no validation messages (`grep -c VUID log`).

### 5. Convergence (still camera)

Capture the same pinned pose at several frames, e.g. `ART_RTIC_STOP_AFTER=hdr ART_RTIC_MAX_FRAMES=4020 ART_RTIC_CAPTURE=dir@300,1000,2000,4000` (release). Then compare consecutive captures with each other, not only with the reference: relative RMS of the per-pixel difference of `E_ind` luminance, and the same on 16×16 block means. A converging image shrinks that difference as \(1/\sqrt{N}\) (from 1000→2000 to 2000→4000 the ratio should be about 0.64). A ratio near 1 means something keeps changing (cache churn, correlated samples); a ratio near 0.5 means a transient decaying as 1/N (fireflies held by a running mean). `_e_ind_frames.pfm` shows how many frames each pixel averages, so history resets are visible directly. Numpy is not installed system-wide; create a venv (`python3 -m venv /tmp/gi/venv && /tmp/gi/venv/bin/pip install numpy pillow`) for such analysis scripts.

### 6. Performance and drift

```bash
env -u LD_PRELOAD ART_RTIC_NO_UI=1 ART_RTIC_BENCH_WANDER=1 ART_RTIC_TIMINGS=1 \
  ART_RTIC_STOP_AFTER=hdr ART_RTIC_MAX_FRAMES=1500 ./target/release/artrtic
```

Do **not** combine with `ART_RTIC_SELFTEST` when reading timings: each readback stalls for up to ~0.8 s and inflates averages. Add a separate wander run with self-tests (e.g. at 300, 900, 1490) to check the pool stays healthy while moving.

### 7. Look at the images

Numbers miss things like moiré, black silhouettes, or blobs. Crop the `_final.jpg`/`_indirect.jpg` captures and compare before/after side by side. Use a new file name for every image you inspect; image viewers (and agent file readers) may cache an image with the same path.

## Current baselines (RTX 4080 Laptop, Sponza, 1920×1080)

Regressions are judged against these. Details in `engine/docs/11-testing.md`.

| Run | Bias | Block RMSE |
|---|---|---|
| default pose, 1 bounce, frames 600/1200 | +0.7% / +0.1% | 23% |
| default pose, multibounce, frames 600/1000/4000 | -0.3% / -0.6% / -1.1% | 15% |
| pose 2, 1 bounce, frames 600/1200 | -0.1% / -0.6% | 8.2% |
| pose 2, multibounce, frames 600/1200 | -1.3% / -1.7% | 8.0% |

- Still-camera convergence (default pose, multibounce): the cache stops changing about 250 frames after the camera stops; frame-to-frame change is 9.2% per pixel (0.97% per block) from 1000→2000 and 6.1% (0.54%) from 2000→4000.
- Wander, unthrottled: GI 6.1–8.7 ms, frame 9.5 ms average, p99 11.3 ms.
- Black pixels in the default view: 311 (0.015%), isolated occluded pockets.
- Self-tests pass; validation layer reports no errors.

## Debugging approach that worked

- **Measure before fixing.** Each issue was first turned into a number (black-pixel count, frame-to-frame RMS, pool counters, worst-pixel coordinates), and each hypothesis was tested with a one-line experiment (e.g. setting a constant to `1e9`, gating a pass) before writing the real fix. Revert experiments immediately.
- **Locate outliers.** For fireflies, list the worst pixels (estimate vs reference) and crop that region; clustered outliers point to a specific surfel or geometric configuration.
- **Bisect the frame** with `ART_RTIC_STOP_AFTER` when a pass hangs or corrupts later passes.
- **Use the debug views** (surfel ids, coverage, indirect, reference, difference) via the UI or `ART_RTIC_VIEW`.
- **Always pin the camera** for measurements. An unpinned window receives mouse and keyboard input, and the camera can move mid-run (it happened: a 4000-frame run silently changed view at frame ~1460).

## Things to watch out for

Shader and build quirks:

- `inline_spirv!` ignores includes (see Build). A shader change that "does nothing" usually was not rebuilt.
- glslang emits invalid SPIR-V for a struct-valued ternary mixing a buffer element and a local (`c ? buf[i] : local`). Use `if`/`else`. `validate.sh` catches it.
- In ray-generation shaders `payload` is overwritten by every trace, including the shadow rays inside `direct_irradiance`. Copy fields into locals right after `trace_closest`.
- The framework's descriptor writer gives each image-array element its own binding slot; new GI bindings go after 27, not after 20.
- Run binaries from the repository root (relative asset paths; running from `engine/` fails, e.g. exit 127 when the binary path is wrong).

Renderer invariants (full list in `engine/docs/12-maintenance.md`):

- Slot ids are stable; only maintain frees, only surfel update and coverage allocate; no pass both pushes and pops the free stack.
- Every allocation bumps `generation`; every free clears the light record.
- The ReSTIR target and its support test must agree: `G ≥ DISK_SUPPORT_FLOOR` wherever `disk_supports` is true, 0 elsewhere (`disk_support_matches_geometry` test).
- The pmf stored in a light tile must be the exact selection probability.
- Any new allocation or free must show up in `ST_FREE`, `ST_EVICTED`, or `ST_DOOMED`, or `ST_STABLE_FRAMES` keeps counting while the cache changes.
- `ST_*` words, strides, grid and hash constants are mirrored in `gi_math.rs`.

Ideas that look right and are wrong (all tried):

- Point-light falloff instead of the exact disk geometry term (off by up to 4× near the disk).
- Dividing by `M` instead of `Z` in ReSTIR (biased dark at concave corners).
- Shrinking the emitting disk to \(\sqrt{A/\pi}\) (error went up).
- Turning the G-buffer normal toward the eye (black silhouettes); orient it by the rasterized face instead.
- A sampler with `max_lod = 0` (only mip 0, moiré).
- Converging a still image by only lengthening the accumulation window (cache churn and temporally correlated reservoirs keep it drifting).
- Switching surfel estimates to running means as soon as the camera stops (newborn surfels' noisy first samples persist for thousands of frames as bright blobs).
- Freezing offscreen spawns as soon as the camera stops (missing offscreen surfels are missing lights: -6.8% bias).

## When you change something

1. Implement, matching the surrounding style. Comments only state constraints the code cannot show.
2. `validate.sh`, touch the embedding Rust files, build release and debug, `cargo test --release --lib`.
3. Self-test plus accuracy gates at both poses, one bounce and multibounce, in the debug build; no validation messages.
4. If the change affects still images, run the convergence check; if it affects cost, the wander timing run.
5. Inspect images for the affected area.
6. Update `engine/docs/` (the chapter for the area, chapter 11 numbers, chapter 12 for new invariants or traps) and the baselines above. Use drawings and schematics when describing data structures and algorithms.
7. Report measured before/after numbers and images, including anything that got worse.

## Hard rule

You DO NOT ignore bugs, you DO NOT postpone fixing them: as soon as you hit a bug that is not in code to be deleted you fix it right away.

The math, algorithm and data structure MUST BE suitable for very large worlds, not just the example Sponza scene. Do not make assumptions on what will be rendered or how big the world is.